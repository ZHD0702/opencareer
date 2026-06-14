from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from services.zhaopin_job_service import ZhaopinSearchError, extract_zhaopin_jobs, score_job


logger = logging.getLogger(__name__)


@dataclass
class BrowserTask:
    id: str
    owner_session_id: str
    query_plan: dict[str, Any]
    status: str = "starting"
    status_text: str = "正在启动浏览器"
    current_url: str = ""
    error: str | None = None
    matches: list[dict[str, Any]] = field(default_factory=list)
    total_candidates: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: BrowserContext | None = None
    page: Page | None = None
    runner: asyncio.Task | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.owner_session_id,
            "query_plan": self.query_plan,
            "status": self.status,
            "status_text": self.status_text,
            "current_url": self.current_url,
            "error": self.error,
            "matches": self.matches,
            "total_candidates": self.total_candidates,
            "created_at": self.created_at,
        }


class BrowserSessionService:
    """Runs Playwright in one dedicated thread, independent of Uvicorn's event loop."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._tasks: dict[str, BrowserTask] = {}
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opencareer-browser")

    async def _in_browser_thread(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    def _ensure_browser_sync(self) -> Browser:
        if self._browser and self._browser.is_connected():
            return self._browser
        self._playwright = sync_playwright().start()
        launch_errors: list[str] = []
        for channel in ("msedge", None, "chrome"):
            try:
                launch_kwargs: dict[str, Any] = {"headless": True, "timeout": 20000}
                if channel:
                    launch_kwargs["channel"] = channel
                self._browser = self._playwright.chromium.launch(**launch_kwargs)
                logger.info("Using Playwright browser: %s", channel or "bundled chromium")
                return self._browser
            except BaseException as exc:
                detail = str(exc).strip() or type(exc).__name__
                launch_errors.append(f"{channel or 'chromium'}: {detail}")
        self._playwright.stop()
        self._playwright = None
        raise RuntimeError("；".join(launch_errors))

    async def _ensure_browser(self) -> Browser:
        async with self._lock:
            return await self._in_browser_thread(self._ensure_browser_sync)

    async def create(self, owner_session_id: str, query_plan: dict[str, Any]) -> dict[str, Any]:
        await self.stop_for_owner(owner_session_id)
        task = BrowserTask(id=str(uuid.uuid4()), owner_session_id=owner_session_id, query_plan=query_plan)
        self._tasks[task.id] = task
        task.runner = asyncio.create_task(self._run_search(task))
        return task.public()

    def get(self, task_id: str, owner_session_id: str | None = None) -> BrowserTask | None:
        task = self._tasks.get(task_id)
        if task and owner_session_id and task.owner_session_id != owner_session_id:
            return None
        return task

    def _run_search_sync(self, task: BrowserTask) -> None:
        browser = self._ensure_browser_sync()
        task.context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            locale="zh-CN",
        )
        task.page = task.context.new_page()
        role = str(task.query_plan.get("role") or "").strip()
        city_code = str(task.query_plan.get("city_code") or "489")
        task.status = "navigating"
        task.status_text = f"正在打开智联招聘并定位{task.query_plan.get('city') or '全国'}岗位"
        task.page.goto(f"https://www.zhaopin.com/sou/jl{city_code}", wait_until="domcontentloaded", timeout=45000)
        task.current_url = task.page.url
        task.page.wait_for_timeout(1600)

        self._continue_search_sync(task)

    def _continue_search_sync(self, task: BrowserTask) -> None:
        if not task.page or task.page.is_closed():
            raise ValueError("浏览器页面尚未就绪")

        page = task.page
        role = str(task.query_plan.get("role") or "").strip()
        city_code = str(task.query_plan.get("city_code") or "489")

        body_text = page.locator("body").inner_text()[:6000]
        if self._is_challenge(body_text, page.url):
            task.status = "needs_user"
            task.status_text = "请先在侧栏完成登录或验证，完成后点击“登录后继续匹配”"
            return

        task.status = "searching"
        task.status_text = f"正在搜索“{role}”"
        search_input = page.locator(
            'input[placeholder*="职位"], input[placeholder*="搜索"], input[type="search"], input[type="text"]'
        ).first
        try:
            search_input.wait_for(state="visible", timeout=5000)
        except Exception:
            page.goto(f"https://www.zhaopin.com/sou/jl{city_code}", wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(1600)
            task.current_url = page.url
            body_text = page.locator("body").inner_text()[:6000]
            if self._is_challenge(body_text, page.url):
                task.status = "needs_user"
                task.status_text = "登录状态仍需确认，请在侧栏处理后再次点击“登录后继续匹配”"
                return
            search_input = page.locator(
                'input[placeholder*="职位"], input[placeholder*="搜索"], input[type="search"], input[type="text"]'
            ).first
            search_input.wait_for(state="visible", timeout=15000)
        search_input.fill(role)
        search_input.press("Enter")
        page.wait_for_timeout(3000)
        task.current_url = page.url
        body_text = page.locator("body").inner_text()[:8000]
        if self._is_challenge(body_text, page.url):
            task.status = "needs_user"
            task.status_text = "搜索触发了验证，请处理后点击“登录后继续匹配”"
            return
        self._update_matches_from_page_sync(task)
        task.status = "ready"
        task.status_text = f"已实时匹配 {len(task.matches)} 条智联岗位"

    def _update_matches_from_page_sync(self, task: BrowserTask, limit: int = 5) -> None:
        if not task.page or task.page.is_closed():
            raise ValueError("浏览器页面尚未就绪")

        source_url = task.page.url
        html = task.page.content()
        try:
            jobs = extract_zhaopin_jobs(html, source_url)
        except ZhaopinSearchError:
            jobs = self._extract_jobs_from_dom_sync(task.page, source_url)

        ranked = [
            score_job(job, task.query_plan)
            for job in jobs
            if job.get("title") and job.get("company") and job.get("url")
        ]
        ranked.sort(
            key=lambda item: (item["match_score"], item.get("published_at") or ""),
            reverse=True,
        )
        task.total_candidates = len(ranked)
        task.matches = ranked[:limit]
        if not task.matches:
            raise ValueError("智联页面已打开，但没有解析到可用岗位")

    @staticmethod
    def _extract_jobs_from_dom_sync(page: Page, source_url: str) -> list[dict[str, Any]]:
        return page.evaluate(
            r"""
            (sourceUrl) => {
              const seen = new Set();
              const results = [];
              const anchors = Array.from(document.querySelectorAll('a[href*="/jobdetail/"]'));
              for (const anchor of anchors) {
                const url = anchor.href || '';
                const title = (anchor.innerText || anchor.textContent || '').trim();
                if (!url || !title || seen.has(url)) continue;
                const card = anchor.closest('article, li, [class*="joblist"], [class*="position"], [class*="job-card"]') || anchor.parentElement?.parentElement;
                const text = (card?.innerText || '').replace(/\s+/g, ' ').trim();
                const companyAnchor = card?.querySelector('a[href*="company"], [class*="company"] a, [class*="company"]');
                const company = (companyAnchor?.innerText || companyAnchor?.textContent || '').trim();
                const salary = (text.match(/(?:\d+(?:\.\d+)?[-~至]\d+(?:\.\d+)?(?:K|k|万|元\/天|元)|薪资面议)/) || [])[0] || '薪资面议';
                const city = (text.match(/北京|上海|广州|深圳|杭州|成都|南京|武汉|西安|苏州|天津|重庆|厦门|长沙|郑州|青岛/) || [])[0] || '';
                if (!company) continue;
                seen.add(url);
                results.push({
                  id: url,
                  title,
                  company,
                  salary,
                  education: '',
                  experience: '',
                  city,
                  district: '',
                  industry: '',
                  company_size: '',
                  financing_stage: '',
                  work_type: text.includes('实习') ? '实习' : '',
                  internship_months: text.includes('实习') ? 1 : 0,
                  weekly_internship_days: 0,
                  skills: [],
                  description: text,
                  published_at: '',
                  url,
                  source_url: sourceUrl,
                });
              }
              return results;
            }
            """,
            source_url,
        )

    async def match_from_existing_session(
        self,
        owner_session_id: str,
        query_plan: dict[str, Any],
        limit: int = 5,
    ) -> dict[str, Any] | None:
        candidates = [
            task
            for task in self._tasks.values()
            if task.owner_session_id == owner_session_id
            and task.status == "ready"
            and task.page is not None
        ]
        if not candidates:
            return None
        task = candidates[-1]
        task.query_plan = query_plan
        await self._in_browser_thread(self._update_matches_from_page_sync, task, limit)
        return {
            "source": "智联招聘",
            "source_mode": "authenticated_browser",
            "total_candidates": task.total_candidates,
            "qualified_candidates": len([job for job in task.matches if job["match_score"] >= 50]),
            "matches": task.matches,
            "warnings": [],
        }

    async def _run_search(self, task: BrowserTask) -> None:
        try:
            await self._in_browser_thread(self._run_search_sync, task)
        except asyncio.CancelledError:
            task.status = "stopped"
            task.status_text = "搜索已停止"
            raise
        except Exception as exc:
            logger.exception("Playwright job search failed")
            task.status = "error"
            task.status_text = "浏览器任务执行失败"
            message = str(exc).strip() or type(exc).__name__
            if "Executable doesn't exist" in message:
                message = "未找到可用浏览器。请安装 Microsoft Edge，或运行：python -m playwright install chromium"
            task.error = message

    async def resume(self, task_id: str) -> dict[str, Any]:
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError("浏览器会话不存在")
        if not task.page or await self._in_browser_thread(task.page.is_closed):
            raise ValueError("浏览器页面已经关闭，请重新匹配岗位")
        if task.runner and not task.runner.done():
            raise ValueError("浏览器任务仍在执行，请稍候")

        task.status = "resuming"
        task.status_text = "正在确认登录状态并继续匹配"
        task.error = None
        task.runner = asyncio.create_task(self._resume_search(task))
        return task.public()

    async def _resume_search(self, task: BrowserTask) -> None:
        try:
            await self._in_browser_thread(self._continue_search_sync, task)
        except asyncio.CancelledError:
            task.status = "stopped"
            task.status_text = "搜索已停止"
            raise
        except Exception as exc:
            logger.exception("Playwright resumed job search failed")
            task.status = "error"
            task.status_text = "继续匹配失败"
            task.error = str(exc).strip() or type(exc).__name__

    @staticmethod
    def _is_challenge(body_text: str, url: str) -> bool:
        text = f"{body_text}\n{url}".lower()
        return any(marker in text for marker in ("安全验证", "验证码", "请完成验证", "verify", "captcha"))

    def _screenshot_sync(self, task: BrowserTask) -> bytes | None:
        if not task.page or task.page.is_closed():
            return None
        task.current_url = task.page.url
        return task.page.screenshot(type="png")

    async def screenshot(self, task_id: str) -> bytes | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        return await self._in_browser_thread(self._screenshot_sync, task)

    def _interact_sync(self, task: BrowserTask, action: dict[str, Any]) -> None:
        if not task.page or task.page.is_closed():
            raise ValueError("浏览器页面尚未就绪")
        page = task.page
        kind = action.get("type")
        if kind == "click":
            page.mouse.click(float(action["x"]), float(action["y"]))
        elif kind == "scroll":
            page.mouse.wheel(0, float(action.get("delta_y", 500)))
        elif kind == "type":
            page.keyboard.type(str(action.get("text") or ""))
        elif kind == "key":
            page.keyboard.press(str(action.get("key") or "Enter"))
        else:
            raise ValueError("不支持的浏览器操作")
        page.wait_for_timeout(350)
        task.current_url = page.url

    async def interact(self, task_id: str, action: dict[str, Any]) -> dict[str, Any]:
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError("浏览器页面尚未就绪")
        await self._in_browser_thread(self._interact_sync, task, action)
        return task.public()

    def _close_task_sync(self, task: BrowserTask) -> None:
        if task.context:
            task.context.close()
            task.context = None
            task.page = None

    async def stop(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.runner and not task.runner.done():
            task.runner.cancel()
        await self._in_browser_thread(self._close_task_sync, task)
        task.status = "stopped"
        task.status_text = "浏览器会话已关闭"
        return True

    async def stop_for_owner(self, owner_session_id: str) -> None:
        for task in list(self._tasks.values()):
            if task.owner_session_id == owner_session_id and task.status not in {"stopped", "error"}:
                await self.stop(task.id)

    def _close_sync(self) -> None:
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    async def close(self) -> None:
        for task_id in list(self._tasks):
            await self.stop(task_id)
        await self._in_browser_thread(self._close_sync)
        self._executor.shutdown(wait=False, cancel_futures=True)


browser_session_service = BrowserSessionService()

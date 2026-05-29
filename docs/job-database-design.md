# 岗位数据库设计方案

### 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：Postgre# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

### 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW()# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
|# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
|# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255)# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired）# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500)# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
|# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20)# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
|# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

|# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2)# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
|# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
|# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at |# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3.# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-oss# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- =================================# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CN# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'full# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_name ON skills USING gin(to_tsvector('chinese', name));

-- =================================# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_name ON skills USING gin(to_tsvector('chinese', name));

-- ============================================
-- 5. 岗位-技能关联表
-- ============================================
CREATE TABLE job_skills (
    id# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_name ON skills USING gin(to_tsvector('chinese', name));

-- ============================================
-- 5. 岗位-技能关联表
-- ============================================
CREATE TABLE job_skills (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    skill# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_name ON skills USING gin(to_tsvector('chinese', name));

-- ============================================
-- 5. 岗位-技能关联表
-- ============================================
CREATE TABLE job_skills (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    importance VARCHAR(20) DEFAULT 'required# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_name ON skills USING gin(to_tsvector('chinese', name));

-- ============================================
-- 5. 岗位-技能关联表
-- ============================================
CREATE TABLE job_skills (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    importance VARCHAR(20) DEFAULT 'required',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, skill_id)
# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_name ON skills USING gin(to_tsvector('chinese', name));

-- ============================================
-- 5. 岗位-技能关联表
-- ============================================
CREATE TABLE job_skills (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    importance VARCHAR(20) DEFAULT 'required',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, skill_id)
);

CREATE INDEX idx_job_skills_job ON job_skills(job_id);
CREATE INDEX idx_job_skills_skill ON job_skills(skill# 岗位数据库设计方案

## 1. 需求分析

### 1.1 功能需求
- **岗位信息存储与查询**：存储岗位名称、公司、薪资、地点等基本信息
- **岗位与简历匹配**：根据用户简历智能推荐匹配岗位
- **岗位数据分析**：薪资趋势、热门岗位、行业分析等

### 1.2 数据来源
- 爬虫从招聘网站抓取岗位数据

### 1.3 技术选型
- **数据库**：PostgreSQL（关系型，支持复杂查询和扩展）

---

## 2. 数据库架构设计

### 2.1 ER图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   companies     │     │     jobs        │     │  job_skills     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ PK id           │◄────┤ PK id           │◄────┤ PK id           │
│    name         │     │ FK company_id   │     │ FK job_id       │
│    industry     │     │    title        │     │ FK skill_id     │
│    size         │     │    description  │     │    importance   │
│    location     │     │    salary_min   │     └─────────────────┘
│    website      │     │    salary_max   │              │
│    logo_url     │     │    salary_type  │              │
│    created_at   │     │    location     │              ▼
└─────────────────┘     │    work_type    │     ┌─────────────────┐
                        │    exp_required │     │    skills       │
                        │    edu_required │     ├─────────────────┤
                        │    status       │     │ PK id           │
                        │    source       │     │    name         │
                        │    source_url   │     │    category     │
                        │    published_at │     │    alias        │
                        │    created_at   │     │    created_at   │
                        └─────────────────┘     └─────────────────┘
                                 │
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ job_applications│
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │ FK user_id      │
                        │    status       │
                        │    applied_at   │
                        │    notes        │
                        │    created_at   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  job_analytics  │
                        ├─────────────────┤
                        │ PK id           │
                        │ FK job_id       │
                        │    views        │
                        │    applications │
                        │    match_score  │
                        │    updated_at   │
                        └─────────────────┘
```

### 2.2 表结构详细设计

#### 2.2.1 companies（公司表）
存储公司基本信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(255) | NOT NULL | 公司名称 |
| industry | VARCHAR(100) | | 所属行业 |
| size | VARCHAR(50) | | 公司规模（如：50-150人） |
| location | VARCHAR(255) | | 公司所在地 |
| website | VARCHAR(255) | | 官网链接 |
| logo_url | VARCHAR(500) | | Logo图片URL |
| description | TEXT | | 公司简介 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.2 jobs（岗位表）
存储岗位核心信息

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| company_id | INTEGER | FOREIGN KEY | 关联公司 |
| title | VARCHAR(255) | NOT NULL | 岗位名称 |
| description | TEXT | | 岗位描述 |
| requirements | TEXT | | 岗位要求 |
| benefits | TEXT | | 福利待遇 |
| salary_min | INTEGER | | 最低薪资 |
| salary_max | INTEGER | | 最高薪资 |
| salary_type | VARCHAR(20) | | 薪资类型（月薪/年薪） |
| currency | VARCHAR(10) | DEFAULT 'CNY' | 货币类型 |
| location | VARCHAR(255) | | 工作地点 |
| work_type | VARCHAR(50) | | 工作类型（全职/兼职/实习） |
| exp_required | VARCHAR(50) | | 经验要求 |
| edu_required | VARCHAR(50) | | 学历要求 |
| status | VARCHAR(20) | DEFAULT 'active' | 状态（active/closed/expired） |
| source | VARCHAR(50) | | 数据来源（如：boss/linkedin） |
| source_url | VARCHAR(500) | | 原始链接 |
| source_id | VARCHAR(100) | | 源站岗位ID（用于去重） |
| published_at | TIMESTAMP | | 发布时间 |
| expires_at | TIMESTAMP | | 过期时间 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

**索引设计**：
```sql
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);
```

#### 2.2.3 skills（技能表）
存储技能标签

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL UNIQUE | 技能名称 |
| category | VARCHAR(50) | | 技能类别（编程语言/框架/工具等） |
| alias | VARCHAR(255) | | 别名（JSON数组） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

#### 2.2.4 job_skills（岗位-技能关联表）
多对多关系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| skill_id | INTEGER | FOREIGN KEY | 关联技能 |
| importance | VARCHAR(20) | DEFAULT 'required' | 重要程度（required/preferred） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

**唯一约束**：(job_id, skill_id)

#### 2.2.5 job_applications（求职申请记录表）
记录用户的求职进度

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY | 关联岗位 |
| user_id | VARCHAR(100) | | 用户ID（关联到现有用户系统） |
| status | VARCHAR(50) | DEFAULT 'saved' | 状态（saved/applied/interviewing/offered/rejected） |
| applied_at | TIMESTAMP | | 申请时间 |
| notes | TEXT | | 备注 |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.6 job_analytics（岗位分析表）
存储岗位的统计和分析数据

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| job_id | INTEGER | FOREIGN KEY UNIQUE | 关联岗位 |
| views | INTEGER | DEFAULT 0 | 浏览次数 |
| applications | INTEGER | DEFAULT 0 | 申请次数 |
| match_score_avg | DECIMAL(3,2) | | 平均匹配分数 |
| salary_percentile | INTEGER | | 薪资百分位 |
| hot_score | DECIMAL(5,2) | | 热度分数 |
| updated_at | TIMESTAMP | DEFAULT NOW() | 更新时间 |

#### 2.2.7 job_categories（岗位分类表）
岗位分类体系

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | SERIAL | PRIMARY KEY | 主键 |
| name | VARCHAR(100) | NOT NULL | 分类名称 |
| parent_id | INTEGER | FOREIGN KEY | 父分类ID（自关联） |
| level | INTEGER | | 层级 |
| keywords | VARCHAR(500) | | 关键词（用于自动分类） |
| created_at | TIMESTAMP | DEFAULT NOW() | 创建时间 |

---

## 3. 数据库初始化脚本

```sql
-- 创建数据库
CREATE DATABASE opencareer_jobs;

-- 连接到新数据库
\c opencareer_jobs;

-- 启用UUID扩展（可选）
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. 公司表
-- ============================================
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    size VARCHAR(50),
    location VARCHAR(255),
    website VARCHAR(255),
    logo_url VARCHAR(500),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_industry ON companies(industry);

-- ============================================
-- 2. 岗位分类表
-- ============================================
CREATE TABLE job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES job_categories(id),
    level INTEGER DEFAULT 1,
    keywords VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_categories_parent ON job_categories(parent_id);

-- ============================================
-- 3. 岗位表
-- ============================================
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES job_categories(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type VARCHAR(20) DEFAULT 'monthly',
    currency VARCHAR(10) DEFAULT 'CNY',
    location VARCHAR(255),
    work_type VARCHAR(50) DEFAULT 'fulltime',
    exp_required VARCHAR(50),
    edu_required VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    source VARCHAR(50),
    source_url VARCHAR(500),
    source_id VARCHAR(100),
    published_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_category ON jobs(category_id);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_id ON jobs(source, source_id);
CREATE UNIQUE INDEX idx_jobs_unique_source ON jobs(source, source_id) WHERE source IS NOT NULL AND source_id IS NOT NULL;
CREATE INDEX idx_jobs_published ON jobs(published_at);
CREATE INDEX idx_jobs_work_type ON jobs(work_type);

-- ============================================
-- 4. 技能表
-- ============================================
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    alias JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_name ON skills USING gin(to_tsvector('chinese', name));

-- ============================================
-- 5. 岗位-技能关联表
-- ============================================
CREATE TABLE job_skills (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    importance VARCHAR(20) DEFAULT 'required',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, skill_id)
);

CREATE INDEX idx_job_skills_job ON job_skills(job_id);
CREATE INDEX idx_job_skills_skill ON job_skills(skill_id);

-- ============================================
-- 6. 求职申请记录表

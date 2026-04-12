# 规范同步更新规则 — UPDATE RULES

> **本文件是强制性工作规范。**
> 每次修改项目中的任何文件之前，必须阅读本文档，并在修改完成后按照规定的检查清单同步更新所有相关文档。
> 本规则同样适用于 Claude Code 在每次会话中的工作。

---

## 核心原则

**一次改动，全量同步。**

代码改动 → 文档不更新 = 文档欠债。
文档欠债积累 → 项目不可维护 → 面试时无法自洽解释。

---

## 必读：每次改动前的 5 步确认

在开始任何改动之前，先回答这 5 个问题：

```
□ 1. 这次改动影响了哪些 src/ 模块？
□ 2. 是否有新增或删除的公共函数/类/常量？
□ 3. 是否改变了用户可见的行为（安装步骤、使用方式、输出格式）？
□ 4. 是否修改了任何超参数或配置项？
□ 5. 这次改动对应哪个版本号变更（patch/minor/major）？
```

---

## 必须同步更新的文档清单

### 每次改动必须检查（全部）

| 文件 | 何时更新 | 更新内容 |
|------|---------|---------|
| `CHANGELOG.md` | **每次改动** | 在对应版本节下添加条目（Added/Changed/Fixed/Removed） |
| `UPDATE_RULES.md` | **每次规范本身改变时** | 保持规范文档自身是最新的 |

### 按改动类型决定是否更新

| 改动类型 | `README.md` | `README_CN.md` | `CONTRIBUTING.md` | `CHANGELOG.md` | Docstring |
|---------|:-----------:|:--------------:|:-----------------:|:--------------:|:---------:|
| 新增公共函数/类 | ✅ | ✅ (同步) | — | ✅ Added | 必须 |
| 修改函数签名 | ✅ | ✅ (同步) | — | ✅ Changed | 必须 |
| 删除函数/类 | ✅ | ✅ (同步) | — | ✅ Removed | N/A |
| 新增 src/ 模块 | ✅ | ✅ (同步) | ✅ 作用域表 | ✅ Added | 必须 |
| 新增依赖项 | ✅ 安装部分 | ✅ (同步) | — | ✅ Added | — |
| 修改超参数 | ✅ | ✅ (同步) | — | ✅ Changed | 必须 |
| Bug 修复（无 API 变化） | — | — | — | ✅ Fixed | 酌情 |
| 新增 Streamlit 页面 | ✅ | ✅ (同步) | — | ✅ Added | — |
| 新增测试 | — | — | — | ✅ Changed | — |
| 修改 CI/CD | — | — | — | ✅ Changed | — |
| 重构（无行为变化） | — | — | — | ✅ Changed | 酌情 |
| 纯文档改动 | ✅ | ✅ | 酌情 | ✅ Changed | — |
| 改动规范/规则文档 | — | — | 酌情 | ✅ Changed | — |

> **双语同步铁律：** `README.md` 和 `README_CN.md` 必须在同一个 commit 内更新。
> 不允许只更新一个。

---

## CHANGELOG.md 更新规则

### 格式规范

```markdown
## [版本号] - YYYY-MM-DD

### Added
- 新增的功能描述（以动词原形开头：Add, Support, Implement...）

### Changed
- 修改的行为描述（以动词原形开头：Update, Rename, Improve...）

### Fixed
- 修复的 Bug 描述（以动词原形开头：Fix, Correct, Resolve...）

### Removed
- 删除的功能（以动词原形开头：Remove, Drop, Delete...）
```

### 版本号规则（语义化版本 Semver）

```
主版本.次版本.补丁版本 （例如 2.2.0）

主版本（Major）：破坏性 API 变化，不向后兼容
次版本（Minor）：新增功能，向后兼容
补丁版本（Patch）：Bug 修复，无新功能
```

| 改动类型 | 版本升级 |
|---------|---------|
| BREAKING CHANGE（API 不兼容） | Major（x.0.0） |
| 新增功能 | Minor（x.y.0） |
| Bug 修复 | Patch（x.y.z） |
| 纯文档更新 | Patch（x.y.z） |

---

## Docstring 更新规则

所有 `src/` 中的公共函数，修改时必须同步更新 docstring：

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """单行摘要（动词原形，不超过 79 字符）。

    可选：更详细的描述段落。

    Args:
        param1: 参数描述。
        param2: 参数描述。默认为 ``default``。

    Returns:
        返回值的类型和描述。

    Raises:
        ValueError: 何时会抛出此异常。

    Example:
        >>> result = function_name("hello", param2=True)
        >>> result
        'expected_output'
    """
```

**检查项：**
- [ ] 函数签名变了 → Args 节必须更新
- [ ] 返回值变了 → Returns 节必须更新
- [ ] 新增异常 → Raises 节必须更新
- [ ] 行为变了 → Example 必须更新

---

## Commit 规范检查

每次提交前确认 commit message 格式：

```
<type>(<scope>): <描述>
```

**type 选项：**
- `feat` — 新功能
- `fix` — Bug 修复
- `docs` — 纯文档
- `refactor` — 重构（无行为变化）
- `test` — 测试
- `chore` — 依赖、配置
- `perf` — 性能优化

**scope 选项（模块名）：**
`config` | `data_loader` | `preprocess` | `baseline_model` | `bert_model` |
`evaluate` | `visualize` | `explain` | `streamlit` | `notebooks` | `docs` |
`deps` | `ci` | `docker`

**禁止的 commit message：**
```
❌ "update code"
❌ "fix bug"
❌ "WIP"
❌ "misc changes"
```

**正确示例：**
```
✅ feat(explain): add shap_to_plotly_bar() for SHAP visualization
✅ fix(data_loader): handle None input in clean_text() without crashing
✅ docs(readme): sync README_CN.md with updated installation steps
```

---

## Claude Code 工作规范（每次会话必读）

> Claude Code 在每次开始工作前，必须按以下顺序操作：

### 开始工作前

```
1. 读取 MEMORY.md → 了解项目当前状态
2. 读取本文件 UPDATE_RULES.md → 确认规范
3. 读取 CLAUDE.md → 确认本地工作规则
4. 浏览要修改的文件 → 理解现有代码
```

### 完成改动后（提交前强制检查）

```
□ CHANGELOG.md 已更新（对应版本节下已添加条目）
□ 如涉及公共 API 变化：README.md 已更新
□ 如 README.md 更新了：README_CN.md 在同一 commit 中也更新了
□ 修改的函数 docstring 已同步更新
□ commit message 符合 Conventional Commits 格式
□ 没有提交 CLAUDE.md、HANDOFF.md、INTERVIEW_GUIDE.md 到 git
□ 没有提交 data/raw/、models/、reports/metrics.json 到 git
□ 更新 memory/project_sentiment_tracker.md（如项目状态有变化）
```

### 绝对不能提交到 GitHub 的文件

```
CLAUDE.md
HANDOFF.md
INTERVIEW_GUIDE.md
data/raw/
data/processed/
models/*.pkl
models/*.pt
reports/metrics.json
notebooks/*_executed.ipynb
.env
```

---

## 本文件的更新规则

本文件（UPDATE_RULES.md）本身也需要维护：

- 如果发现规则有遗漏 → 立即补充，并在 CHANGELOG.md 中记录
- 如果某条规则不再适用 → 删除并说明原因
- 如果项目新增了新类型的文件 → 在同步表格中加入该文件
- **每次更新本文件后，在 CHANGELOG.md 中添加一条 `docs(rules)` 条目**

---

## 快速参考卡（打印贴在显示器旁）

```
改了代码？→ 改 CHANGELOG.md（必须）
改了 API？→ 改 README.md + README_CN.md（同一 commit）
改了函数？→ 改 Docstring
改了规范？→ 改 UPDATE_RULES.md + CONTRIBUTING.md
提交前？→ 检查 5 步清单
推送前？→ 确认不包含私人文件
```

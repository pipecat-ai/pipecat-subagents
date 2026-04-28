---
name: update-docs
description: Update documentation pages to match source code changes on the current branch
---

Update documentation pages to reflect source code changes on the current branch. Analyzes the diff against main, maps changed source files to their corresponding doc pages, and makes targeted edits.

## Arguments

```
/update-docs [DOCS_PATH]
```

- `DOCS_PATH` (optional): Path to the docs repository root. If not provided, ask the user.

Examples:
- `/update-docs /Users/me/src/docs`
- `/update-docs`

## Instructions

### Step 1: Resolve docs path

If `DOCS_PATH` was provided as an argument, use it. Otherwise, ask the user for the path to their docs repository.

Verify the path exists and contains `subagents/` and `api-reference/pipecat-subagents/` subdirectories.

### Step 2: Create docs branch

Get the current pipecat-subagents branch name:
```bash
git rev-parse --abbrev-ref HEAD
```

In the docs repo, create a new branch off main with a matching name:
```bash
cd DOCS_PATH && git checkout main && git pull && git checkout -b {branch-name}-docs
```

For example, if the pipecat-subagents branch is `feat/new-feature`, the docs branch becomes `feat/new-feature-docs`.

All doc edits in subsequent steps are made on this branch.

### Step 3: Detect changed source files

Run:
```bash
git diff main..HEAD --name-only
```

Filter to files that could affect documentation:
- `src/pipecat_subagents/agents/**/*.py` (agent implementations)
- `src/pipecat_subagents/bus/**/*.py` (bus, bridge, messages, serializers)
- `src/pipecat_subagents/runner/**/*.py` (agent runner)
- `src/pipecat_subagents/registry/**/*.py` (agent registry)
- `src/pipecat_subagents/types.py` (shared types)

Ignore `__init__.py`, `__pycache__`, test files, `clowder/**`, and `bus/queue.py`.

### Step 4: Map source files to doc pages

For each changed source file, find the corresponding doc page. Read the mapping file at `.claude/skills/update-docs/SOURCE_DOC_MAPPING.md` and look up the file in the direct mapping table. If not found, use the search fallback.

### Step 5: Analyze each source-doc pair

For each mapped pair:

1. **Read the full source file** to understand current state
2. **Read the diff** for that file: `git diff main..HEAD -- <source_file>`
3. **Read the current doc page** in full

Identify what changed by comparing source to docs:

- **Constructor parameters**: Compare `__init__` signature to the Configuration section's `<ParamField>` entries
- **Properties**: Compare `@property` definitions to the Properties section
- **Methods**: Compare method signatures and docstrings to the Methods section
- **Dataclass fields**: Compare field definitions to the types/messages field tables
- **Decorator parameters**: Compare decorator function signatures to the Decorators page
- **Class names / imports**: Check if usage examples reference correct names
- **Behavioral changes**: Check if descriptions or notes need updating

### Step 6: Make targeted edits

For each doc page that needs updates, edit **only the sections that need changes**. Preserve all other content exactly as-is.

#### Rules

- **Never remove content** unless the corresponding source code was removed
- **Never rewrite sections** that are already accurate
- **Match existing formatting** -- if the page uses `<ParamField>` tags, use them; if it uses tables, use tables
- **Keep descriptions concise** -- match the tone and length of surrounding content
- **Preserve CardGroup, links, and examples** unless they reference removed functionality
- **Don't touch frontmatter** unless a class was renamed

#### Section-specific guidance

**Configuration** (constructor params):
- Use `<ParamField path="name" type="type" default="value">` format if the page already uses it
- Add new params in logical order (required first, then optional)
- Remove params that no longer exist in source
- Update types/defaults that changed

**Properties**:
- Format is `### prop_name` heading + code block showing `agent.prop -> Type` + description paragraph
- Compare to `@property` definitions in the source file
- Add new properties, remove deleted ones, update types or descriptions that changed

**Methods**:
- Format is `### method_name` heading + code block with full method signature + description + parameter table (`| Parameter | Type | Default | Description |`)
- Compare method signatures and docstrings from source
- Add new public methods, remove deleted ones, update signatures and descriptions
- Don't document private methods (prefixed with `_`)

**Bus messages** (messages.mdx):
- Overview table organized by category, then per-message field tables
- Compare to dataclass fields in `bus/messages.py`
- Add new message types, remove deleted ones, update fields

**Types** (types.mdx):
- Dataclass heading + import example + field table
- Compare to dataclass definitions in `types.py`, `task_context.py`, and `registry/registry.py`
- Add new types, remove deleted ones, update fields

**Decorators** (decorators.mdx):
- `<ParamField>` entries for decorator parameters + usage examples
- Compare to decorator function signatures in `agents/llm/tool_decorator.py`, `task_decorator.py`, `watch_decorator.py`
- Update parameter types, defaults, and descriptions

**Usage** (code examples):
- Update import paths, class names, and parameter names
- Only modify examples if they would break or be misleading with the new API
- Don't rewrite working examples just to add new optional params

**Notes**:
- Add notes for new behavioral gotchas or breaking changes
- Remove notes about limitations that were fixed
- Keep existing notes that are still accurate

**Overview / Key Features / Prerequisites**:
- Only update if the PR fundamentally changes what a component does (new capability, removed capability, renamed class)
- Most PRs will NOT need changes to these sections

### Step 7: Update guides

Guides reference specific class names, parameters, imports, and code patterns. After completing reference doc edits, check if any guides need updates too.

For each changed source file, collect the class names, renamed parameters, and changed imports from the diff. Search the guide directories:
```bash
grep -rl "ClassName\|old_param_name" DOCS_PATH/subagents/
```

For each guide that references changed code:
1. Read the full guide
2. Update class names, parameter names, import paths, and code examples that are now incorrect
3. **Don't rewrite prose** -- only fix the specific references that changed
4. Leave guides alone if they reference the component generally but don't use any changed APIs

Guide directories:
- `subagents/learn/` -- learning guides (agents, handoff, tasks, flows, distributed, proxy, etc.)
- `subagents/fundamentals/` -- fundamentals pages (agent bus, registry, bus bridge)
- `subagents/examples/` -- example overview

### Step 8: Identify doc gaps

After processing all mapped pairs, check for gaps:

**Missing pages**: Source files that had no doc page mapping and are not on the skip list. For each, report:
- The source file path
- The main class(es) it defines
- Whether a new doc page should be created

**Missing sections**: Mapped doc pages that are missing standard sections compared to the source. Flag these for manual doc creation.

Do NOT auto-create new doc pages. Subagents doc pages have varied structures (e.g., `base-agent.mdx` has 8+ sections, `decorators.mdx` has 3 independent sections). Auto-generated pages would be inconsistent. Flag gaps and let a human create new pages.

Do NOT modify `docs.json` or any navigation files. Subagents has a stable nav structure. New components should be flagged for manual addition.

### Step 9: Output summary

After all edits are complete, print a summary:

```
## Documentation Updates

### Updated reference pages
- `api-reference/pipecat-subagents/base-agent.mdx` -- Updated Methods (added `new_method`), Properties (updated `active` type)
- `api-reference/pipecat-subagents/decorators.mdx` -- Updated @tool section (added `parallel` param)

### Updated guides
- `subagents/learn/agents-and-runner.mdx` -- Updated code example (renamed `old_param` to `new_param`)

### Unmapped source files
- `src/pipecat_subagents/agents/new_thing.py` -- NewThing class (no doc page exists)

### Skipped files
- `src/pipecat_subagents/clowder/agent.py` -- observability tool, no docs coverage
```

## Guidelines

- **Be conservative** -- only change what the diff warrants. Don't "improve" docs beyond what changed in source.
- **Read before editing** -- always read the full doc page before making changes so you understand the existing structure.
- **Preserve voice** -- match the writing style of the existing doc page, don't impose a different tone.
- **One PR at a time** -- this skill operates on the current branch's diff against main. Don't look at other branches.
- **Parallel analysis** -- when multiple source files map to different doc pages, analyze and edit them in parallel for efficiency.

## Checklist

Before finishing, verify:

- [ ] All changed source files were checked against the mapping table
- [ ] Each doc page edit matches the actual source code change (not guessed)
- [ ] No content was removed unless the corresponding source was removed
- [ ] New parameters have accurate types and defaults from source
- [ ] Formatting matches the existing page style
- [ ] Guides referencing changed APIs were checked and updated
- [ ] Unmapped files were reported to the user

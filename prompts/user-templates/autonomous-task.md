# Autonomous Task Template

Use this template for roadmaps that the harness will execute autonomously.

## Usage

1. Copy this template to your project
2. Fill in the placeholders
3. Run with `python -m src.um_agent_coder.harness --roadmap roadmap.md --autonomous`

---

# [PROJECT NAME] Roadmap

## Objective

[Describe the high-level goal in 1-2 sentences]

## Constraints

- max_time: [e.g., 8h, 24h]
- max_iterations: [e.g., 500, unlimited]
- working_directory: [e.g., ./]

## Success Criteria

- [ ] [Criterion 1: e.g., All tests pass]
- [ ] [Criterion 2: e.g., Coverage > 80%]
- [ ] [Criterion 3: e.g., No linting errors]

## Tasks

### Task 1: [Task Name]

- [ ] **task-001**: [Detailed description of what to accomplish]
  - timeout: [e.g., 30min]
  - depends: [e.g., none, task-000]
  - success: [How to verify this task is complete]
  - cli: [codex | gemini | claude | auto]
  - ralph: [true | false]
  - max_iterations: [e.g., 50]
  - completion_promise: [e.g., TASK_001_COMPLETE]

  [Optional: Additional context or instructions for the agent]

### Task 2: [Task Name]

- [ ] **task-002**: [Description]
  - timeout: 30min
  - depends: task-001
  - success: [Verification criteria]
  - cli: auto

### Task 3: [Task Name]

- [ ] **task-003**: [Description]
  - timeout: 30min
  - depends: task-002
  - success: [Verification criteria]
  - cli: auto

## Growth Mode

[Optional: Tasks to generate after roadmap completes]

- Improve test coverage
- Optimize performance
- Add documentation

---

## Example: Filled Template

```markdown
# Authentication System Roadmap

## Objective

Implement a complete JWT-based authentication system for the FastAPI backend.

## Constraints

- max_time: 8h
- max_iterations: 500
- working_directory: ./backend

## Success Criteria

- [ ] All auth tests pass
- [ ] No security vulnerabilities (bandit scan)
- [ ] API endpoints documented in OpenAPI

## Tasks

### Task 1: User Model

- [ ] **auth-001**: Create User model with SQLAlchemy
  - timeout: 30min
  - depends: none
  - success: User model exists with email, password_hash fields
  - cli: codex
  - ralph: true
  - max_iterations: 30
  - completion_promise: USER_MODEL_COMPLETE

  Requirements:
  - Use SQLAlchemy 2.0 syntax
  - Include created_at, updated_at timestamps
  - Email must be unique

### Task 2: Password Hashing

- [ ] **auth-002**: Implement password hashing with bcrypt
  - timeout: 20min
  - depends: auth-001
  - success: hash_password and verify_password functions work
  - cli: codex

### Task 3: Login Endpoint

- [ ] **auth-003**: Create POST /auth/login endpoint
  - timeout: 45min
  - depends: auth-001, auth-002
  - success: Returns JWT on valid credentials, 401 on invalid
  - cli: codex
  - ralph: true
  - max_iterations: 50
  - completion_promise: LOGIN_ENDPOINT_COMPLETE

### Task 4: Protected Routes

- [ ] **auth-004**: Implement JWT middleware for protected routes
  - timeout: 30min
  - depends: auth-003
  - success: Protected routes return 401 without valid JWT
  - cli: codex
```

---

## Tips

1. **Be specific**: Vague goals lead to poor results
2. **Use ralph mode**: For complex tasks that need iteration
3. **Set realistic timeouts**: Allow time for the agent to explore
4. **Define clear success criteria**: How will you know it's done?
5. **Use dependencies**: Order tasks logically
6. **Route appropriately**: Use gemini for research, codex for implementation

---

*Template for um-agent-coder harness*

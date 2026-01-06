# Research Then Build Template

Use this template for tasks that require understanding before implementation. Uses Gemini for research, then Codex for implementation.

## Usage

1. Copy this template
2. Fill in the research questions and implementation tasks
3. Run with `python -m src.um_agent_coder.harness --roadmap roadmap.md --autonomous`

---

# [FEATURE NAME] Roadmap

## Objective

[What are you building?]

## Phase 1: Research

### Understand Requirements

- [ ] **research-001**: Research [topic/technology/pattern]
  - timeout: 30min
  - depends: none
  - success: Summary document exists at docs/research/[topic].md
  - cli: gemini

  Questions to answer:
  - [Question 1]
  - [Question 2]
  - [Question 3]

  Output format:
  - Summary of findings
  - Recommended approach
  - Potential challenges
  - Code examples if applicable

### Explore Existing Code

- [ ] **research-002**: Analyze existing codebase patterns
  - timeout: 20min
  - depends: none
  - success: Pattern analysis at docs/research/patterns.md
  - cli: gemini

  Focus areas:
  - How similar features are implemented
  - Testing patterns used
  - Error handling conventions

## Phase 2: Design

### Architecture Decision

- [ ] **design-001**: Create implementation design
  - timeout: 30min
  - depends: research-001, research-002
  - success: Design document at docs/design/[feature].md
  - cli: claude

  Design should include:
  - Component diagram
  - Data flow
  - API contracts
  - Testing strategy

## Phase 3: Implementation

### Core Implementation

- [ ] **impl-001**: Implement [component 1]
  - timeout: 45min
  - depends: design-001
  - success: [Component] works, tests pass
  - cli: codex
  - ralph: true
  - max_iterations: 50
  - completion_promise: COMPONENT_1_COMPLETE

### Integration

- [ ] **impl-002**: Integrate with existing system
  - timeout: 30min
  - depends: impl-001
  - success: Integration tests pass
  - cli: codex

### Polish

- [ ] **impl-003**: Add documentation and cleanup
  - timeout: 20min
  - depends: impl-002
  - success: Docstrings added, no linting errors
  - cli: codex

---

## Example: Adding OAuth Support

```markdown
# OAuth Integration Roadmap

## Objective

Add Google OAuth login to the existing authentication system.

## Phase 1: Research

### Understand OAuth 2.0

- [ ] **research-001**: Research OAuth 2.0 flow and Google implementation
  - timeout: 30min
  - depends: none
  - success: docs/research/oauth.md exists
  - cli: gemini

  Questions to answer:
  - What is the OAuth 2.0 authorization code flow?
  - What scopes do we need for user profile?
  - How do we handle token refresh?
  - What libraries are available for Python?

### Analyze Current Auth

- [ ] **research-002**: Analyze current authentication implementation
  - timeout: 20min
  - depends: none
  - success: docs/research/current-auth.md exists
  - cli: gemini

  Focus:
  - How JWT tokens are currently generated
  - How user sessions are managed
  - Where auth middleware is defined

## Phase 2: Design

### OAuth Design

- [ ] **design-001**: Design OAuth integration
  - timeout: 30min
  - depends: research-001, research-002
  - success: docs/design/oauth.md exists
  - cli: claude

  Design decisions:
  - Use authlib or python-social-auth?
  - Store Google tokens or just use for initial auth?
  - How to link Google accounts to existing users?

## Phase 3: Implementation

### Google OAuth Endpoint

- [ ] **impl-001**: Implement /auth/google/login endpoint
  - timeout: 45min
  - depends: design-001
  - success: Endpoint redirects to Google consent screen
  - cli: codex
  - ralph: true
  - max_iterations: 40
  - completion_promise: GOOGLE_LOGIN_COMPLETE

### Callback Handler

- [ ] **impl-002**: Implement /auth/google/callback endpoint
  - timeout: 45min
  - depends: impl-001
  - success: Callback creates/links user, returns JWT
  - cli: codex
  - ralph: true
  - max_iterations: 40
  - completion_promise: GOOGLE_CALLBACK_COMPLETE

### UI Integration

- [ ] **impl-003**: Add "Login with Google" button to frontend
  - timeout: 30min
  - depends: impl-002
  - success: Button visible, redirects to Google
  - cli: codex

### Testing

- [ ] **impl-004**: Add integration tests for OAuth flow
  - timeout: 30min
  - depends: impl-003
  - success: Tests pass with mocked Google responses
  - cli: codex
```

---

## CLI Routing Strategy

| Phase | CLI | Why |
|-------|-----|-----|
| Research | gemini | Large context, good at exploration |
| Design | claude | Strong reasoning, architecture |
| Implementation | codex | Fast code generation |
| Review | claude | Quality judgment |

## Tips

1. **Research thoroughly first**: Better understanding = better implementation
2. **Document findings**: Research docs help if you need to revisit
3. **Use Claude for design**: Complex decisions benefit from strong reasoning
4. **Use Codex for code**: Fast, good at implementation
5. **Ralph mode for complex tasks**: Let it iterate until done

---

*Template for research-driven development with um-agent-coder*

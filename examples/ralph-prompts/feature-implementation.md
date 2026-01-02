# Ralph Prompt Template: Feature Implementation

Use this template for implementing new features with the Ralph Loop.

## Roadmap Task Definition

```markdown
- [ ] **feat-XXX**: [Feature description]. Output <promise>FEATURE_NAME_COMPLETE</promise> when all requirements are met.
  - ralph: true
  - max_iterations: 30
  - completion_promise: FEATURE_NAME_COMPLETE
  - timeout: 60min
  - success: [Specific, measurable success criteria]
  - cli: codex
  - depends: [any dependencies]
```

## Example: User Authentication

```markdown
- [ ] **feat-001**: Implement JWT-based user authentication. Output <promise>AUTH_FEATURE_COMPLETE</promise> when login, logout, and token refresh all work with tests.
  - ralph: true
  - max_iterations: 35
  - completion_promise: AUTH_FEATURE_COMPLETE
  - timeout: 60min
  - success: POST /api/login returns JWT, POST /api/logout invalidates token, POST /api/refresh returns new token, minimum 10 unit tests, all tests pass
  - cli: codex
  - depends: none
```

## Example: Payment Integration

```markdown
- [ ] **feat-002**: Integrate Stripe payment processing. Output <promise>STRIPE_INTEGRATION_COMPLETE</promise> when payment flow works end-to-end.
  - ralph: true
  - max_iterations: 40
  - completion_promise: STRIPE_INTEGRATION_COMPLETE
  - timeout: 90min
  - success: Create checkout session, handle webhook events (payment_intent.succeeded, payment_intent.failed), store transaction records, integration tests with Stripe test mode
  - cli: codex
  - depends: feat-001
```

## Example: Search Functionality

```markdown
- [ ] **feat-003**: Add full-text search for products. Output <promise>SEARCH_FEATURE_COMPLETE</promise> when search returns relevant results with pagination.
  - ralph: true
  - max_iterations: 25
  - completion_promise: SEARCH_FEATURE_COMPLETE
  - timeout: 45min
  - success: GET /api/search?q=term returns matching products, supports pagination (limit/offset), relevance scoring, minimum 8 unit tests
  - cli: codex
  - depends: none
```

## Tips for Feature Tasks

1. **Be Specific**: Include exact API endpoints, data formats, and behaviors
2. **Include Tests**: Always require tests as part of success criteria
3. **Set Realistic Iterations**: Features typically need 20-40 iterations
4. **Use Longer Timeouts**: Features are complex; 45-90min per iteration is common
5. **Specify Dependencies**: Ensure prerequisite features are completed first

## Common Success Criteria Patterns

```markdown
# API endpoint
- success: [METHOD] /api/[path] returns [response], handles [errors], has [N] unit tests

# Database migration
- success: Migration creates [tables/columns], rollback works, data preserved

# UI component
- success: Component renders [elements], handles [interactions], has [N] tests

# Background job
- success: Job processes [items], handles failures gracefully, has integration tests
```

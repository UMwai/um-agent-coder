# Roadmap: Example SaaS Project

## Objective
Create a SaaS application generating $500 MRR through a subscription-based service.

## Constraints
- Max time per task: 30 min
- Max retries per task: 3
- Working directory: ./project

## Success Criteria
- [ ] Website deployed and accessible
- [ ] User authentication working
- [ ] Payment processing functional
- [ ] At least one paying customer

## Tasks

### Phase 1: MVP Setup
- [ ] **task-001**: Initialize Next.js 14 project with TypeScript and Tailwind CSS
  - timeout: 15min
  - depends: none
  - success: `npm run build` passes without errors
  - cwd: ./project

- [ ] **task-002**: Set up authentication with Clerk or NextAuth
  - timeout: 30min
  - depends: task-001
  - success: Users can sign up, log in, and log out

- [ ] **task-003**: Create basic dashboard layout with navigation
  - timeout: 20min
  - depends: task-002
  - success: Dashboard accessible after login

### Phase 2: Core Features
- [ ] **task-004**: Implement the core SaaS feature/value proposition
  - timeout: 45min
  - depends: task-003
  - success: Core feature works end-to-end

- [ ] **task-005**: Add data persistence with database (Supabase/Postgres)
  - timeout: 30min
  - depends: task-004
  - success: Data persists across sessions

### Phase 3: Monetization
- [ ] **task-006**: Integrate Stripe for payment processing
  - timeout: 30min
  - depends: task-005
  - success: Test payment completes successfully

- [ ] **task-007**: Create pricing page with subscription tiers
  - timeout: 20min
  - depends: task-006
  - success: Pricing page displays correctly

- [ ] **task-008**: Implement subscription management (upgrade/downgrade/cancel)
  - timeout: 30min
  - depends: task-007
  - success: Users can manage their subscription

### Phase 4: Launch
- [ ] **task-009**: Deploy to Vercel with production configuration
  - timeout: 20min
  - depends: task-008
  - success: Site accessible at production URL

- [ ] **task-010**: Set up monitoring and error tracking (Sentry)
  - timeout: 15min
  - depends: task-009
  - success: Errors are logged and alerts work

## Growth Mode
When all phases complete, Codex will autonomously:
1. Analyze the deployed application for improvements
2. Optimize performance and user experience
3. Add features that could increase conversion
4. Improve SEO and discoverability
5. Enhance error handling and reliability
6. Continue iterating until stopped

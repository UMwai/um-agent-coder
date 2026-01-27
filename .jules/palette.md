## 2026-01-27 - Turning Errors into Onboarding
**Learning:** Users often run CLI tools without arguments to see what happens. Treating this as an error is standard, but treating it as an invitation (interactive mode) reduces friction and feels more welcoming.
**Action:** When a required argument is missing in a TTY context, consider prompting for it interactively instead of exiting.

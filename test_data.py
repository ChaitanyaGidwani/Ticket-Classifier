"""28 labeled test messages (20 single-label, 8 multi-issue) + duplicate pairs."""
T = lambda text, labels, urgent=False: {"text": text, "labels": labels, "urgent": urgent}

TEST_SET = [
    # Billing
    T("I see two identical payments on my bank statement for the same order.", ["Billing"]),
    T("Could you explain the extra fee on my latest invoice?", ["Billing"]),
    T("My refund still hasn't arrived after two weeks.", ["Billing"]),
    T("The coupon code didn't apply at checkout and I paid full price.", ["Billing"]),
    # Technical Support
    T("The mobile app freezes whenever I try to open the settings page.", ["Technical Support"]),
    T("I keep getting a timeout error when loading my reports.", ["Technical Support"]),
    T("Videos won't play in my browser, I only see a blank screen.", ["Technical Support"]),
    T("After updating, the app shows a white screen on launch.", ["Technical Support"]),
    # Security
    T("There were login attempts on my account from an IP address I don't recognize.", ["Security"]),
    T("Is your platform GDPR compliant and how is customer data stored?", ["Security"]),
    T("I clicked a link in a strange email and now I'm worried my credentials leaked.", ["Security"]),
    T("Can you tell me how to turn on multi-factor login?", ["Security"]),
    # Cancellation
    T("Please end my subscription before the next billing date.", ["Cancellation"]),
    T("Our team is shutting down, so we need to close our account.", ["Cancellation"]),
    T("I want to opt out of the annual plan and stop the auto-renewal.", ["Cancellation"]),
    T("Kindly process my cancellation request for the enterprise plan.", ["Cancellation"]),
    # Feature Request
    T("Would be amazing if the app could sync with Google Calendar.", ["Feature Request"]),
    T("Any plans to support custom themes?", ["Feature Request"]),
    T("It'd really help to have a way to export data as PDF.", ["Feature Request"]),
    T("Please add keyboard shortcuts for common actions.", ["Feature Request"]),
    # Urgent singles
    T("URGENT: our payment failed and the service will be suspended today.", ["Billing"], True),
    T("Just a quick suggestion: add a search bar to the history page.", ["Feature Request"]),
    # Multi-issue
    T("I was charged twice this month. Also, the app keeps crashing on startup.", ["Billing", "Technical Support"]),
    T("My account looks hacked and I see unauthorized charges on my card. Please help immediately!", ["Security", "Billing"], True),
    T("I want to cancel my subscription, but first I'd like a refund for the last payment.", ["Cancellation", "Billing"]),
    T("The login page throws an error. It would also be great to have a dark mode.", ["Technical Support", "Feature Request"]),
    T("Our production dashboard is down and we need this fixed ASAP. Plus, can you add Slack alerts?", ["Technical Support", "Feature Request"], True),
    T("Someone changed my password and I can't log in anymore. I'm also being billed for a plan I never bought.", ["Security", "Billing"], True),
]

# (ticket A, ticket B, is_duplicate)
DUP_PAIRS = [
    ("I was charged twice this month", "My card got billed two times for this month", True),
    ("The app crashes when I open it", "Every time I launch the application it crashes", True),
    ("Please cancel my subscription", "I would like to terminate my plan", True),
    ("I was charged twice this month", "The app crashes when I open it", False),
    ("Please add dark mode", "How do I enable two-factor authentication?", False),
    ("I want a refund", "Can you add an Android app?", False),
]

"""Held-out test and validation dataset for Support Ticket Classifier.

Contains 170 independently curated and labeled support tickets completely separated
from seeds in classifier.py.
Includes:
- Single-issue tickets across all 5 classes
- Multi-issue tickets (2 categories)
- Hard cases: short/ambiguous phrases, typos, informal English/Hinglish
- Hard negatives (e.g. 'cancel my card' -> Billing, 'film production' -> Feature Request)
- Urgent tickets (both true positives and tricky false-positive traps)
- Split into VALIDATION_SET (for threshold tuning & calibration) and TEST_SET_HELDOUT (for final unbiased evaluation)
- HELDOUT_DUP_PAIRS for semantic duplicate evaluation
"""

T = lambda text, labels, urgent=False: {"text": text, "labels": labels, "urgent": urgent}

ALL_HELDOUT_DATA = [
    # ==================== BILLING (SINGLE) ====================
    T("Can I pay via wire transfer instead of credit card?", ["Billing"]),
    T("My bank says the payment went through but your system says unpaid.", ["Billing"]),
    T("Why is there an international transaction tax on my invoice?", ["Billing"]),
    T("I got billed $49 instead of the discounted $29 student rate.", ["Billing"]),
    T("Please update our company billing address for tax compliance.", ["Billing"]),
    T("How do I switch from monthly to annual billing cycle?", ["Billing"]),
    T("My corporate card expired and I need to update payment details.", ["Billing"]),
    T("We need a consolidated invoice for all 15 team seats.", ["Billing"]),
    T("The annual renewal fee was processed without any advance warning email.", ["Billing"]),
    T("Can I get an extension on the invoice due date?", ["Billing"]),
    T("Where do I find our VAT number on past billing statements?", ["Billing"]),
    T("I was charged twice on my Amex for transaction #9021.", ["Billing"]),
    T("Card declined because of 3D Secure verification failure.", ["Billing"]),
    T("Can you provide a W-9 form before we issue payment?", ["Billing"]),
    T("Can you cancel my card on file? I do not want it stored in your database.", ["Billing"]),  # Hard negative for Cancellation
    T("Cancel the pending invoice #4088, we will pay via direct wire.", ["Billing"]),  # Hard negative for Cancellation
    T("refudn for overcharge not recieved yet after five days", ["Billing"]),  # Typo
    T("paise kat gaye credit card se but subscription activate nahi hui", ["Billing"]),  # Hinglish
    T("mera balance deduct ho gaya par plan renew nahi hua", ["Billing"]),  # Hinglish
    T("extra charges on my receipt that make no sense at all", ["Billing"]),
    T("Do you accept PayPal or crypto for enterprise billing?", ["Billing"]),
    T("I need an itemized breakdown of compute charges on my bill.", ["Billing"]),
    T("Why was my discount coupon revoked after month one?", ["Billing"]),
    T("Disputed charge: bank asked for merchant clarification on invoice 552.", ["Billing"]),
    T("Failed recurring payment notification received today on card ending 4012.", ["Billing"]),
    T("Where can I download the receipt for the March 12 payment?", ["Billing"]),
    T("billing query", ["Billing"]),  # Short / ambiguous
    T("Why did the sales tax percentage change on our invoice?", ["Billing"]),
    T("Can we split our subscription payment across two debit cards?", ["Billing"]),
    T("We were billed during our free trial period in error.", ["Billing"]),
    T("Payment gateway returned error code 1042 card issuer unavailable.", ["Billing"]),
    T("How do I request a formal proforma invoice for procurement?", ["Billing"]),

    # ==================== TECHNICAL SUPPORT (SINGLE) ====================
    T("WebSocket connection drops every 5 minutes in Safari.", ["Technical Support"]),
    T("The API endpoint /v1/sync returns 502 Bad Gateway intermittently.", ["Technical Support"]),
    T("CSV file export results in a corrupted 0-byte download.", ["Technical Support"]),
    T("Push notifications are delayed by over 3 hours on iOS 17.", ["Technical Support"]),
    T("Getting CORS policy block error when calling the webhooks from localhost.", ["Technical Support"]),
    T("Search index seems stuck, new documents do not show up in query results.", ["Technical Support"]),
    T("Cannot view PDF attachments inside the browser viewer.", ["Technical Support"]),
    T("The mobile app crashes immediately after scanning a QR code.", ["Technical Support"]),
    T("Database sync spinner spins continuously without finishing.", ["Technical Support"]),
    T("Dropdown menu does not open on Firefox touch screen devices.", ["Technical Support"]),
    T("Got Error: ECONNRESET when fetching large workspace data.", ["Technical Support"]),
    T("Screen turns completely black after maximizing the chart widget.", ["Technical Support"]),
    T("Cant log in, it keeps saying gateway timeout 504", ["Technical Support"]),  # Typo
    T("app bar bar crash ho raha hai kholte hi", ["Technical Support"]),  # Hinglish
    T("login page pe ghoomta rehta hai aage nahi badhta", ["Technical Support"]),  # Hinglish
    T("sound notifications not playing on Chrome 124", ["Technical Support"]),
    T("The rich text editor deletes bold formatting when pasting from external editor.", ["Technical Support"]),
    T("Tables with more than 1000 rows cause the browser tab to run out of memory.", ["Technical Support"]),
    T("Session tokens expire after only 5 minutes of inactivity.", ["Technical Support"]),
    T("Keyboard shortcut Ctrl+K no longer opens the quick command palette.", ["Technical Support"]),
    T("Image uploads hang at 99% and fail with a network error.", ["Technical Support"]),
    T("Cancel my support ticket #8812, the issue resolved itself after restart.", ["Technical Support"]),  # Hard negative for Cancellation
    T("Internal server error 500 when saving user profile preferences.", ["Technical Support"]),
    T("system not working properly", ["Technical Support"]),  # Short / ambiguous
    T("technical glitch", ["Technical Support"]),  # Short / ambiguous
    T("The desktop app window remains completely blank after waking from sleep.", ["Technical Support"]),
    T("CPU utilization spikes to 100% when rendering large SVG graphics.", ["Technical Support"]),
    T("OAuth redirect URI mismatch error when logging in via GitHub.", ["Technical Support"]),
    T("Spell check is not functioning inside note descriptions.", ["Technical Support"]),
    T("Auto-save is failing with localized error 409 conflict.", ["Technical Support"]),

    # ==================== SECURITY (SINGLE) ====================
    T("We received a flood of password reset emails that nobody on our team requested.", ["Security"]),
    T("Can we enforce SSO with Okta / SAML for all team accounts?", ["Security"]),
    T("How do we generate an audit trail report of admin user actions?", ["Security"]),
    T("Where can I find your SOC2 Type II compliance certificate and report?", ["Security"]),
    T("I noticed an unknown API key was generated under my developer profile.", ["Security"]),
    T("An ex-employee still has access to our shared workspace, please revoke all sessions.", ["Security"]),
    T("Can you enforce mandatory hardware security key (FIDO2 / YubiKey) login?", ["Security"]),
    T("Is data at rest encrypted using AES-256 or customer-managed KMS keys?", ["Security"]),
    T("We want to conduct a third-party penetration test against our staging environment.", ["Security"]),
    T("A phishing link was posted inside our public customer workspace channel.", ["Security"]),
    T("suspicius activity on my acount from unexpected IP address", ["Security"]),  # Typo
    T("mera password kisi ne change kar diya mujhe suspicious alert aaya", ["Security"]),  # Hinglish
    T("kya hamara customer data encrypted hai server pe?", ["Security"]),  # Hinglish
    T("How do I configure IP allowlisting for enterprise API access?", ["Security"]),
    T("How do I delete old API tokens that are no longer in use?", ["Security"]),
    T("Is your infrastructure protected against distributed denial-of-service (DDoS)?", ["Security"]),
    T("Can we set a custom password expiration policy for team members?", ["Security"]),
    T("Security notification says new session started from Linux device in Berlin.", ["Security"]),
    T("How secure is your refund authorization process against employee fraud?", ["Security"]),  # Hard negative (contains refund)
    T("privacy and data safety inquiry", ["Security"]),
    T("security audit question", ["Security"]),  # Short / ambiguous
    T("Our security scanner detected outdated OpenSSL libraries in your client build.", ["Security"]),
    T("Can we restrict employee logins strictly to our office corporate VPN CIDR blocks?", ["Security"]),
    T("We require all admin accounts to rotate passwords every 60 days.", ["Security"]),
    T("How is personally identifiable customer information scrubbed from your logs?", ["Security"]),
    T("Can we configure custom session timeouts for sensitive financial workflows?", ["Security"]),

    # ==================== CANCELLATION (SINGLE) ====================
    T("Please terminate our enterprise contract at the end of the current quarter.", ["Cancellation"]),
    T("We are switching to a competitor and want to export our data before closing.", ["Cancellation"]),
    T("How do I stop auto-renewal so my card won't be charged next year?", ["Cancellation"]),
    T("Our startup ran out of runway, please discontinue our paid organization plan.", ["Cancellation"]),
    T("I want to delete my profile and all associated personal information permanently.", ["Cancellation"]),
    T("Kindly cancel our pilot subscription and confirm our account is deactivated.", ["Cancellation"]),
    T("Downgrade our team tier to the free plan immediately.", ["Cancellation"]),
    T("I have decided to retire and no longer need this software suite.", ["Cancellation"]),
    T("Where is the cancel subscription button in the new dashboard layout?", ["Cancellation"]),
    T("Please close my workspace and purge all stored files from your cloud.", ["Cancellation"]),
    T("I want to unsubscribe from all marketing and transactional product emails.", ["Cancellation"]),
    T("cancell my account pls, not using anymore", ["Cancellation"]),  # Typo
    T("mujhe account close karna hai, renewal band karo", ["Cancellation"]),  # Hinglish
    T("I suspended my own account temporarily while on vacation.", ["Cancellation"]),  # Tricky urgency negative (contains 'suspended')
    T("We are sunsetting our product line so we need to terminate our vendor agreement.", ["Cancellation"]),
    T("Please remove our domain from your platform and terminate the license.", ["Cancellation"]),
    T("Cancel my pro membership before the upcoming renewal date.", ["Cancellation"]),
    T("I want to end my free trial early so it does not convert to paid.", ["Cancellation"]),
    T("terminate our contract", ["Cancellation"]),  # Short / ambiguous
    T("close account please", ["Cancellation"]),  # Short / ambiguous
    T("Our project has concluded, please discontinue our workspace plan.", ["Cancellation"]),
    T("I am leaving this organization and want my user account removed.", ["Cancellation"]),
    T("Kindly refund the remaining balance and cancel our recurring license.", ["Cancellation", "Billing"]),
    T("Please terminate our contract effective the last day of this month.", ["Cancellation"]),
    T("How do I delete an unused sub-account within our family subscription?", ["Cancellation"]),

    # ==================== FEATURE REQUEST (SINGLE) ====================
    T("Could you add support for webhook signatures so we can verify request origins?", ["Feature Request"]),
    T("We really need a native dark mode theme to reduce eye strain.", ["Feature Request"]),
    T("Is there any plan to support Portuguese and Spanish localization?", ["Feature Request"]),
    T("Would love an integration with Linear or Jira for issue synchronization.", ["Feature Request"]),
    T("Can you provide an option to schedule automated weekly email summary reports?", ["Feature Request"]),
    T("A mobile companion app for iPad with Apple Pencil support would be fantastic.", ["Feature Request"]),
    T("Please add Markdown formatting support inside table cells.", ["Feature Request"]),
    T("Can we get bulk tag management and mass editing across multiple records?", ["Feature Request"]),
    T("Would be awesome if you added role-based granular permissions for viewers vs editors.", ["Feature Request"]),
    T("Can you allow exporting analytics data directly to Google BigQuery or Snowflake?", ["Feature Request"]),
    T("Please add 2FA via SMS as an alternative to authenticator apps.", ["Feature Request"]),
    T("It would help a lot if we could customize the logo on generated invoices.", ["Feature Request"]),
    T("new featrue: dark mode kab aayega app me?", ["Feature Request"]),  # Typo + Hinglish
    T("excel export ka option chahiye dashboard me reports ke liye", ["Feature Request"]),  # Hinglish
    T("I am working on a film production and would love video export support.", ["Feature Request"]),  # Tricky urgency negative (contains 'production')
    T("Can I schedule messages before a deadline?", ["Feature Request"]),  # Tricky urgency negative (contains 'deadline')
    T("We have a soft deadline next quarter for our team trial.", ["Feature Request"]),  # Tricky urgency negative (contains 'deadline')
    T("Custom color themes and brand fonts would be a great addition to the editor.", ["Feature Request"]),
    T("Please support biometric FaceID unlock on iOS devices.", ["Feature Request"]),
    T("Would be nice to have audit log export to AWS S3 buckets.", ["Feature Request"]),
    T("feature idea", ["Feature Request"]),  # Short / ambiguous
    T("suggestion for app", ["Feature Request"]),  # Short / ambiguous
    T("Can we have an auto-dark-mode toggle based on the operating system setting?", ["Feature Request"]),
    T("Please add support for nested sub-folders in the document organization view.", ["Feature Request"]),
    T("Would love a Kanban board view option in addition to the standard table view.", ["Feature Request"]),
    T("Can you add a custom webhook event when a customer status changes?", ["Feature Request"]),
    T("An automated spell-check and grammar suggestion tool would be super useful.", ["Feature Request"]),

    # ==================== DELIVERY ISSUE (SINGLE) ====================
    T("I want to reschedule the delivery of my order to tomorrow afternoon.", ["Delivery Issue"]),
    T("The tracking link shows delivered but no package was left at my front door.", ["Delivery Issue"]),
    T("Can I change the delivery address for order #8841 before it dispatches?", ["Delivery Issue"]),
    T("My shipment arrived in a damaged box and the contents inside are broken.", ["Delivery Issue"]),
    T("Estimated delivery was three days ago and the package is still delayed in transit.", ["Delivery Issue"]),
    T("Courier was unable to access my building gate, how do I schedule redelivery?", ["Delivery Issue"]),
    T("I need expedited express shipping or overnight delivery for my parcel.", ["Delivery Issue"]),
    T("Wrong item was delivered in my parcel, I received someone else's order.", ["Delivery Issue"]),
    T("delivery reschedule karni hai kal ke liye time slot change", ["Delivery Issue"]),  # Hinglish
    T("mera parcel deliver nahi hua tracking shows stuck", ["Delivery Issue"]),  # Hinglish
    T("package delivery issue", ["Delivery Issue"]),  # Short / ambiguous
    T("Customs clearance holds are delaying my international package arrival.", ["Delivery Issue"]),

    # ==================== MULTI-ISSUE TICKETS (2 LABELS) ====================
    T("I was charged twice on my credit card; also the mobile app is crashing on launch.", ["Billing", "Technical Support"]),
    T("My invoice amount is wrong, and I also want to cancel my renewal at month end.", ["Billing", "Cancellation"]),
    T("We see fraudulent charges on our card and suspect our admin credentials leaked.", ["Security", "Billing"], True),
    T("The login page throws a 500 error, and could you also add a dark mode option?", ["Technical Support", "Feature Request"]),
    T("Please cancel my subscription, but first refund the unauthorized renewal fee.", ["Cancellation", "Billing"]),
    T("Someone hacked into our system and now our production API is completely down!", ["Security", "Technical Support"], True),
    T("Can you fix the invoice download button? It gives an error when generating the PDF.", ["Billing", "Technical Support"]),
    T("I want to delete my account permanently and also get a receipt for my final payment.", ["Cancellation", "Billing"]),
    T("The export to Excel feature fails with a timeout error every time.", ["Feature Request", "Technical Support"]),
    T("We need SSO authentication support, and also how do I update our corporate payment method?", ["Security", "Billing"]),
    T("I want to close my account because the app keeps lagging and dropping data.", ["Cancellation", "Technical Support"]),
    T("Can you add Slack alerts? Plus our webhook notifications stopped firing yesterday.", ["Feature Request", "Technical Support"]),
    T("I think our credentials were stolen in a phishing attack, and someone changed our billing email.", ["Security", "Billing"], True),
    T("Our database is locked up after updating, and why did our monthly price increase?", ["Technical Support", "Billing"]),
    T("I would like to request a refund for the downtime, and please cancel my plan.", ["Billing", "Cancellation"]),
    T("A competitor accessed our workspace data; emergency outage on all webhooks right now.", ["Security", "Technical Support"], True),
    T("The mobile app won't open, and I also cannot find where to upgrade to enterprise.", ["Technical Support", "Billing"]),
    T("Payment was deducted two times, aur app bar bar crash ho raha hai.", ["Billing", "Technical Support"]),  # Hinglish multi
    T("Account hack ho gaya hai unauthorized charges bhi dikh rahe hain turant help karo!", ["Security", "Billing"], True),  # Hinglish multi urgent
    T("Please terminate our subscription. Also, is customer data permanently deleted after cancellation?", ["Cancellation", "Security"]),
    T("Can you add multi-currency support? Also our latest transaction in Yen was declined.", ["Feature Request", "Billing"]),
    T("The system freezes during CSV upload; would also love a progress bar feature.", ["Technical Support", "Feature Request"]),
    T("I want to report an unauthorized login from unknown IP, and please lock my account immediately.", ["Security", "Technical Support"], True),
    T("Refund my last payment because the software is broken and completely unusable.", ["Billing", "Technical Support"]),
    T("Our subscription was canceled unexpectedly even though our payment details are valid.", ["Cancellation", "Billing"]),
    T("Dark mode is missing, and the navigation bar overlaps text on smaller screens.", ["Feature Request", "Technical Support"]),
    T("Unknown charges appeared on our company card after our developer lost his laptop.", ["Security", "Billing"], True),
    T("We want to unsubscribe from the annual plan, and can we export our project history to CSV first?", ["Cancellation", "Feature Request"]),
    T("Urgent: payment failure caused immediate service suspension during our product launch!", ["Billing", "Technical Support"], True),
    T("Can you add automated PDF invoices? Also our latest bill shows an incorrect company name.", ["Feature Request", "Billing"]),
    T("I forgot my 2FA backup codes and cannot log in, plus I need my latest billing statement.", ["Security", "Billing"]),
    T("The search filter is broken on Android, and please add voice search support.", ["Technical Support", "Feature Request"]),
    T("Our card was billed after we asked to terminate our contract last month.", ["Billing", "Cancellation"]),
    T("There is a cross-site scripting bug in the comment box, and can you add a markdown preview?", ["Security", "Feature Request"]),
    T("We are canceling our contract because your security audit documentation is missing.", ["Cancellation", "Security"]),
    T("My package never arrived, and I want a full refund credited to my card.", ["Delivery Issue", "Billing"]),
    T("The parcel was delivered damaged. Also the mobile app crashed when opening camera.", ["Delivery Issue", "Technical Support"]),

    # ==================== URGENT TICKETS & TRICKY FALSE POSITIVES ====================
    T("CRITICAL: Our production database is down and all customer traffic is failing!", ["Technical Support"], True),
    T("Emergency: Hacker is currently transferring funds from our business account!", ["Security"], True),
    T("URGENT OUTAGE: API returning 503 across US-East region right now!", ["Technical Support"], True),
    T("All customer passwords were dumped online in a data breach alert!", ["Security"], True),
    T("Double charge of $10,000 processed in error, overdrafting our bank account immediately!", ["Billing"], True),
    T("Production environment is completely inaccessible during our holiday launch sale!", ["Technical Support"], True),
    T("Immediate assistance needed: customer data is leaking publicly on our site!", ["Security"], True),
    T("Emergency: all checkout payments are failing across our entire store right now!", ["Billing", "Technical Support"], True),
    T("I have a tight production deadline for my university video film project.", ["Technical Support"], False),  # Contains 'production' & 'deadline' but NOT urgent
    T("Our creative team is suspended in disbelief at how fast the new update is!", ["Feature Request"], False),  # Contains 'suspended' but NOT urgent
    T("We scheduled a routine maintenance deadline for next month's sprint.", ["Technical Support"], False),  # Contains 'deadline' but NOT urgent
    T("Is there a lawsuit or legal disclosure section in your terms of service?", ["Security"], False),  # Contains 'lawsuit' in informational question
    T("Can you help me avoid late payment suspension by extending our grace period?", ["Billing"], False),  # Contains 'suspension' but routine inquiry
]

# Deterministic Stratified-style split into VALIDATION_SET and TEST_SET_HELDOUT:
# Alternating elements ensures identical class balance, multi-issue ratio, and hard cases across both sets.
VALIDATION_SET = ALL_HELDOUT_DATA[0::2]   # 101 tickets
TEST_SET_HELDOUT = ALL_HELDOUT_DATA[1::2] # 101 tickets

# Held-out duplicate detection benchmark pairs (24 pairs)
# Formatted as: (ticket_a, ticket_b, is_duplicate)
HELDOUT_DUP_PAIRS = [
    # True duplicates (paraphrases / near-duplicates)
    ("Can I pay via wire transfer instead of credit card?", "Do you accept bank wire transfer payments rather than cards?", True),
    ("The mobile app crashes immediately after scanning a QR code.", "Scanning any QR code causes the phone app to crash on launch.", True),
    ("Please terminate our enterprise contract at the end of the current quarter.", "We want to cancel and terminate our quarterly enterprise agreement.", True),
    ("We really need a native dark mode theme to reduce eye strain.", "Please add dark mode theme support to make it easier on the eyes.", True),
    ("An ex-employee still has access to our shared workspace, please revoke all sessions.", "Revoke active sessions for a former staff member who still has access to our account.", True),
    ("I was charged twice on my Amex for transaction #9021.", "My American Express card was double charged for the same transaction.", True),
    ("Push notifications are delayed by over 3 hours on iOS 17.", "Notifications take 3+ hours to arrive on my iPhone running iOS 17.", True),
    ("How do I stop auto-renewal so my card won't be charged next year?", "I want to turn off annual auto-renewal to prevent future card charges.", True),
    ("Is there any plan to support Portuguese and Spanish localization?", "Any plans for multi-language translation in Spanish and Portuguese?", True),
    ("We received a flood of password reset emails that nobody on our team requested.", "Spam password reset requests are arriving that our employees never triggered.", True),
    ("I want to reschedule the delivery of my order to next Monday.", "Can I change my order delivery date and reschedule shipment?", True),
    ("The courier tracking shows delivered but my parcel never arrived.", "My package tracking says delivered, however I never received the shipment.", True),

    # Hard negatives (surface similarity or shared keywords, but DIFFERENT semantics)
    ("Can you cancel my card on file? I do not want it stored in your database.", "Please cancel my subscription and close my account at the end of the month.", False),
    ("I suspended my own account temporarily while on vacation.", "Our account was suspended because our recurring payment failed.", False),
    ("I am working on a film production and would love video export support.", "Our production system is completely down and inaccessible right now!", False),
    ("We have a soft deadline next quarter for our team trial.", "Critical outage: our platform is broken and we have an executive deadline in one hour!", False),
    ("Can you provide a W-9 form before we issue payment?", "I need an itemized breakdown of compute charges on my bill.", False),
    ("WebSocket connection drops every 5 minutes in Safari.", "Dropdown menu does not open on Firefox touch screen devices.", False),
    ("Can we enforce SSO with Okta / SAML for all team accounts?", "Can we set a custom password expiration policy for team members?", False),
    ("We are switching to a competitor and want to export our data before closing.", "Can you allow exporting analytics data directly to Google BigQuery or Snowflake?", False),
    ("The rich text editor deletes bold formatting when pasting from external editor.", "Keyboard shortcut Ctrl+K no longer opens the quick command palette.", False),
    ("Do you accept PayPal or crypto for enterprise billing?", "How secure is your refund authorization process against employee fraud?", False),
    ("I want to cancel the delivery of my package before it ships.", "Please cancel my recurring monthly subscription.", False),
    ("Can I track the real-time courier location for my delivery?", "How do I track usage and query count on our billing dashboard?", False),
]

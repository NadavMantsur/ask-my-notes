# Project Atlas

Atlas is Lumen’s original **customer billing platform**. It calculates invoices from usage events, applies contracts, and pushes PDFs to customers. If someone says “billing” without a version number, they still mean this system — not the rewrite.

## History and status

Atlas **launched March 12, 2024**. The internal codename during stealth was **Nimbus**. Please keep using “Atlas” in tickets; Nimbus only appears in old design docs and in this paragraph so people can search for it.

The product is in **maintenance mode**. We fix billing correctness bugs and tax-table updates. We do not add features. New usage-based work belongs on Atlas v2 (see project-atlas-v2.md). If a salesperson promises a feature on “Atlas,” confirm which system they mean before you estimate.

## Stack and ownership

- Language: Python 3.11
- Database: **PostgreSQL** (do not point Atlas jobs at Mongo)
- Lead: **Maya Chen**
- Slack: #atlas
- War room: 4B (see wifi-and-office for the map)

The Postgres schema is the source of truth for invoice line items. There is a read replica for analytics. Never run migrations on the replica.

## How it relates to v2

Atlas v2 is a separate codebase with a similar name and a launch date that is easy to mix up (theirs is also a March date, one year later). Atlas and Atlas v2 do **not** share a database. Copying a connection string from one page to the other will corrupt invoices. If you are new, your default assignment is this repo (onboarding.md), not v2.

## Runbooks

PagerDuty: Atlas-prod. After-hours, Maya is primary, rotating secondary is posted in #atlas. The Friday demo still includes a two-minute Atlas status even though the v2 team presents longer.

# Project Atlas v2

Atlas v2 is the **usage-based billing rewrite**. It replaces the original Atlas invoice pipeline with event-level pricing. It is not a new version of the same repo. If you joined to work on “Atlas” and your manager is Jordan, you are here. If your manager is Maya, you want project-atlas.md instead.

## History and status

Atlas v2 **launched March 15, 2025**. That date is one year and three days after original Atlas. People mix them up constantly. When you write “Atlas launched in March,” always include the year and the product name.

The migration project name is **Horizon**. Horizon tickets track cutting a customer over from Atlas Postgres to v2. Horizon is not a third product; it is the cutover checklist.

## Stack and ownership

- Language: TypeScript
- Database: **MongoDB** (do not point v2 services at Atlas Postgres)
- Lead: **Jordan Okonkwo**
- Slack: #atlas-v2
- War room: 4C, not 4B

Mongo collections are named after billing concepts (`invoices`, `rate_cards`, `usage_windows`). There is no ORM layer that will save you from writing to the wrong cluster. Double-check the connection string.

## Do not mix the two Atlases

Original Atlas still serves customers who have not been through Horizon. Those invoices live in PostgreSQL under Maya’s team. v2 invoices live in Mongo. A “quick join” across the two databases is forbidden. If a dashboard needs both, go through the approved export, not ad-hoc queries.

## Demo and rituals

Atlas v2 presents at the shared Friday demo (see team-rituals.md) after the two-minute Atlas maintenance update. Bring a customer-ready clip, not a localhost screenshot.

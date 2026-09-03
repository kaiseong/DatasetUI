# Phase 13 — Delivery and V1 Acceptance

All delivery methods require a successful `export_gate` result for the exact current
dataset fingerprint. A later metadata revision invalidates that gate automatically.

- **NAS Export** copies into `exports/.incoming-*`, verifies the complete tree, and
  atomically renames it into the requested final name.
- **Hugging Face** uses a server-only write token, fixes the namespace to
  `rainbowrobotics`, defaults to private, creates new repositories only, adds lineage
  to a staging README, and removes a newly-created remote repository if upload fails.
- **Ubuntu PC / key** is a durable I/O job using a server-registered key and strict
  known-host verification.
- **Ubuntu PC / one-time password** runs directly in the API process. The password is
  held only in the request and SSH client memory; it is never written to SQLite,
  Redis, job JSON, files, logs, or process arguments. A server restart fails that
  request rather than persisting or retrying the credential.

PC copies accept private-network IP addresses only, restrict destinations to the
user's home directory, reject existing destinations, upload to a hidden incoming
directory, and rename only after every file is transferred.

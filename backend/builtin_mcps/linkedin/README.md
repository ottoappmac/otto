# LinkedIn (built-in MCP)

Gives Otto LinkedIn as you: job search, profiles, companies, your feed and your
inbox. It runs the third-party, Apache-2.0
[`mcp-server-linkedin`](https://github.com/stickerdaniel/linkedin-mcp-server)
package, pinned and hash-locked in `requirements.txt`. The package drives its
own Chromium window (Patchright) signed in to your LinkedIn account.

> **Account risk.** LinkedIn's User Agreement does not allow bots or automated
> access, and it restricts accounts it catches. Keep use light and read-mostly:
> a few searches a day, no bulk scraping, no auto-applying, no message or
> connection campaigns, and no unattended schedules.

## Turning it on and signing in

1. On the Agents page, Tools tab, find **LinkedIn (your browser session)** and
   click **Start**. It ships disabled, so nothing opens at app launch. The
   first start builds its venv and downloads its Chromium (about 370 MB).
2. Ask Otto for something on LinkedIn, for example "search LinkedIn jobs for
   product manager roles in Sydney posted this week".
3. The first call that needs your account opens a **LinkedIn sign-in window**.
   Sign in there yourself, including any code or app confirmation LinkedIn
   asks for. Otto never sees or types your password.
4. Once your feed shows, tell Otto to carry on. The session is remembered
   across restarts until LinkedIn signs you out, and then the same window
   comes back.

`get_session_status` reports whether a session is stored and whether
LinkedIn has flagged the account (`account_restricted`).

## What stays on your Mac

* **Where the session lives.** Everything is under
  `~/Library/Application Support/Otto/mcp_server/linkedin/`:
  * `state/profile/` is the Chromium profile holding your LinkedIn cookies.
    Your password is not stored. Anything that can read this folder can use
    your session, so treat it like a saved login.
  * `state/patchright-browsers/` is the downloaded Chromium.
  * `home/` holds the package's own caches.
* **Signing out.** Stop the server in Otto, then delete the `state/` folder.
* **Network.** It only contacts LinkedIn, plus the one-time Chromium download.
  `server.py` forces off the two features that contact the package author's
  servers:
  * the remote tool catalog (`LINKEDIN_MCP_REMOTE_TOOLS=off`), which would
    add tools that run on `mcp.daniel.sticker.name`;
  * the daily update check (`LINKEDIN_MCP_CHECK_FOR_UPDATES=off`).

  `--no-auto-import` stops it from copying cookies out of your everyday
  browser. One exception: when a page fails to parse, the package's error
  report runs a GitHub issue search using the section name and runtime label.
* **No Seatbelt profile.** Unlike most built-ins, this server is not wrapped
  in Otto's per-MCP sandbox, because Chromium cannot run inside it. See
  `UNSANDBOXED_SERVER_IDS` in `backend/action_policy/catalog.py`. That matches
  Otto's Playwright browser.

## Tools and approvals

Classified in `backend/action_policy/catalog.py`:

| Asks first | Tools |
|---|---|
| Send message (`message_send`) | `send_message`, `reply_to_conversation`, `connect_with_person` |
| Read messages (`message_read`) | `get_inbox`, `get_conversation`, `search_conversations` |
| Delete or modify (`destructive`) | `archive_conversation`, `dismiss_job` |
| Drive the browser (`browser_automation`) | `get_easy_apply_form` (reads the form, never submits) |

Not gated: `search_jobs`, `get_job_details`, `get_job_apply_url`,
`get_saved_jobs`, `get_person_profile`, `search_people`, `get_my_profile`,
`get_sidebar_profiles`, `get_company_profile`, `get_company_posts`,
`get_company_employees`, `search_companies`, `get_feed`, `search_posts`,
`get_session_status`, `close_session`.

Sends are outbound, so after Otto has read LinkedIn content in a run they
ignore session grants and always show a card.

## Bumping the version

1. Pick a release at least two weeks old.
2. Regenerate `requirements.txt` with the command in its header.
3. List the new release's tools, update `EXPECTED_TOOLS` in
   `tests/test_linkedin_mcp.py` and the catalog to match, and check any new
   environment flags that phone home.

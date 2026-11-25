"""App cards: how to work an app on macOS, returned by the desktop agent's ``app_guide`` tool.

A card holds facts about the app, the kind its documentation or a few minutes with it would give: its process
name, whether it can be scripted, where common commands live in its menus, how it saves, and what it cannot do.
Cards never hold anything taken from a benchmark task or its checker (a file, a value, a URL from a task).

Both the app's agent and the benchmark agent get cards through the tool, so neither needs file access to read a
skill. Menu paths are for the English UI of current versions; the agent is told to confirm a path with
``menu_search`` when a ``menu_select`` does not find it.
"""

from __future__ import annotations

from .macos_ax_struct import norm_label

_MENU_TOOLS = (
    "Menus: choose items with `menu_select(app, 'Menu > Item')`; find an item you cannot place with "
    "`menu_search(app, 'word')`. Do not write System Events menu scripts for this app."
)

LIBREOFFICE = f"""\
# LibreOffice (Calc, Writer, Impress)

- Shown as "LibreOffice"; its process is `soffice` (System Events: `tell process "soffice"`). One app holds every
  open document; each document is its own window ("<file> - LibreOffice Calc").
- No AppleScript dictionary. Work through its menus and dialogs. {_MENU_TOOLS}
- Dialogs (Format Cells, Chart, Sort, Paragraph…) open as separate windows: read them with `read_dialog`, set
  several fields and press OK with one `fill_dialog`.
- The document on disk is the version from before your edits. Never change the open file with another program:
  LibreOffice keeps its own copy and writes it over yours when it saves.

## Calc
- Go to or select cells with the Name Box (the field left of the formula bar, labelled "Name Box"): type `B2`,
  `A1:G12` or `D2:D30`, then Return; the result says which cells are now selected. Then type into the selection:
  a value or a formula (`=SUM(B2:B11)`), Return moves down, Tab moves right.
- Enter a block of values or formulas in one call with `write_cells(app, 'B2', [[...], [...]])`: one inner list
  per row. It reads the cells back and lists any that differ (Calc's AutoInput can complete a typed word).
- Before writing, read the header row and how far the data goes (`read_table`): write each result under the
  header that names it, in the rows the data has, and fill every row (a total row or column ends with its
  corner cell).
- Fill a formula down: select the range starting at the cell that has it, then Sheet > Fill Cells > Fill Down (⌘D).
- Read what is on screen in one call with `read_table(app)` (only the cells shown are in the tree; select or
  scroll for more). It labels each column and row: the first column shown is not always A, so write to the
  letters it shows.
- Sorting: Data > Sort… sorts the selected cells only. Select the whole table (every column, or one cell inside
  it, which LibreOffice extends to the table) so rows stay together.
- Data > Text to Columns… writes the pieces into the selected column and the ones to its right, over what is
  there: insert empty columns first when the next columns hold data, or copy the column somewhere free.
- A number format's code is in Format > Cells… > Numbers > Format code: `#,##0.00`, `0%`, `0.0,"K"` (each
  trailing comma divides by 1000), `[RED]` colours, `;;;` hides. Check the field reads exactly your code.
- Common menus: Format > Cells… (⌘1, number formats and decimals); Format > Merge and Unmerge Cells > Merge Cells;
  Format > Rows > Hide; Sheet > Insert Sheet…; Sheet > Rename Sheet…; Sheet > Move or Copy Sheet…;
  Data > Sort…; Data > AutoFilter; Data > Text to Columns…; Data > Validity…; Insert > Chart…;
  Insert > Pivot Table…; View > Zoom > Zoom….
- Freezing: View > Freeze Rows and Columns freezes the rows above and the columns left of the selected cell
  (select B3 to keep rows 1–2 and column A in view; to keep the range A1:B1 in view, i.e. row 1 and columns
  A–B, select C2: one row below and one column right of the range's last cell). It is a toggle: choosing it while a freeze is on removes
  the freeze, so to move a freeze choose it twice (off, then on with the new cell selected). View > Freeze Cells
  has only Freeze First Row / Column.
- A table of totals or counts per category (per product, per region, per month) built from rows of data is a
  pivot table: select the data, Insert > Pivot Table… (Data > Pivot Table > Insert or Edit… in older versions),
  keep "Current selection", then drag fields (with `drag`, from the field's name to the box) to Row Fields,
  Column Fields and Data Fields and press OK. It is put on a new sheet unless you choose a destination. SUMIF
  formulas give the same numbers but are not a pivot table.
- A sheet made with Sheet > Insert Sheet… (or the + tab) gets the next default name (Sheet2, Sheet3…). Keep it
  unless a name is asked for: do not rename a sheet to describe its contents.
- A new sheet goes after the existing ones: the Insert Sheet dialog defaults to "Before current sheet", so pick
  "After current sheet" (or use Sheet > Insert Sheet at End…). A check compares sheets by position, and a new sheet
  in front of the original moves the original to second place.
- Sheet tabs are usually not in the controls list: switch sheets with Sheet > Navigate > Go to Sheet… (or
  ⌥⌘PageDown / ⌥⌘PageUp), and rename with Sheet > Rename Sheet….
- Print scaling ("fit to one page", "fit to width") is in Format > Page Style… > Sheet > Scale, pop-up
  "Scaling mode". PDF: File > Export as > Export as PDF….
- Typing into a cell: select it (Name Box or click), then type with `type_text`; the Name Box itself only takes
  cell references.
- Insert > Chart… uses the selected range and opens the Chart Wizard (Chart Type, Data Range, Data Series,
  Chart Elements, where the title goes; then Finish). Select the data first.
- Saving: ⌘S. For an .xlsx or .csv file LibreOffice may ask which format to keep (unless that warning is turned
  off): if a dialog is open after saving, press "Use Excel 2007-365!" (or the button naming the file's own
  format). Do not leave that dialog open.

## Writer
- Select exact text with `select_text(app, 'the words')` (or `paragraph=True` for its whole paragraph) instead of
  clicking: it says what got selected. Before formatting, check the selection is exactly the text the task
  names (the status bar also shows "N words, M characters selected"). Clicks land mid-paragraph, and ⇧End
  selects to the end of the line, ⇧⌘↓ to the end of the document.
- Undo (⌘Z) goes back one step at a time, including earlier steps that were right: read again after each undo.
- Find & Replace (⌘H): "Regular expressions" makes `\n` in Replace a paragraph break and `$` a paragraph end;
  "Current selection only" limits Replace All to the selection; Find All selects every match at once.
- Font colour, highlighting and other toolbar drop-downs are not in the controls list: use Format > Character…
  (Font Effects tab: Font color) or Format > Paragraph….
- Common menus: Format > Character…; Format > Paragraph… (spacing, alignment, tabs); Format > Text > (case
  changes); Insert > Page Number…; Insert > Page Break; Insert > Field; Table > Insert Table… (⌘F12);
  Table > Convert > Text to Table…; File > Export as > Export as PDF….
- Numbered or bracketed lists ("1.", "[1]") are usually automatic numbering: the number is not text. To add an
  entry, put the cursor at the end of the list's last entry, press Return and type only the entry's text; a new
  paragraph elsewhere does not continue the list.
- A citation that points to a list entry, heading or caption is a cross-reference: Insert > Cross-reference…
  (Type: Numbered Paragraphs, Headings or the caption category; Refer using: Number), so it updates with the list.
  Type over a placeholder only after selecting exactly the placeholder (Find & Replace selects it).
- Default fonts for new documents: LibreOffice > Settings… (Preferences… in older versions, ⌘,) >
  LibreOffice Writer > Basic Fonts (Western).
- Saving: ⌘S; for a .docx file keep the format with "Use Word 2007-365!".
"""

GIMP = f"""\
# GIMP (3.x)

- Shown as "GIMP"; the process may be named `gimp` or `GIMP` (observe() shows it). No AppleScript dictionary.
- {_MENU_TOOLS}
- Search every command by name: click the image canvas, press `/` ("Search Actions"), type a word, Return. If
  nothing opens, the canvas did not have focus.
- Image > Mode offers RGB, Grayscale and Indexed. GIMP edits raster images; it does not edit video, and it
  cannot trace a raster image into vector shapes or save one as SVG.
- What GIMP cannot do: open camera RAW files without a RAW loader plug-in (darktable or RawTherapee; without one
  it says no loader is installed), process a folder of files in one go (there is no batch command), or say
  what a picture shows (it has no image recognition).
- Number fields with a slider (opacity, size, the values in Colors dialogs) take typed numbers: click the number,
  select all, type, Return. `set_text_at(x, y, '40')` does that in one call. Dragging the slider is imprecise.
- Moving a layer by an exact amount: Layer > Transform > Offset… (⇧⌘O); the Move tool needs `drag` on the canvas.
- The foreground and background colours open a colour dialog with an "HTML notation" field (hex, e.g. ff0000).
- Common menus: Image > Mode > Indexed…; Image > Transform > Flip Horizontally; Image > Canvas Size…;
  Image > Scale Image…; Layer > New Layer… (⇧⌘N); Layer > Transparency > Add Alpha Channel;
  Colors > Brightness-Contrast…; Colors > Hue-Saturation…; Colors > Curves…; Colors > Color to Alpha…;
  Filters > Light and Shadow > Vignette…; Select > By Color.
- Color and filter commands open a dialog with an OK button: set it with `fill_dialog`, then OK. Nothing is
  applied until OK is pressed.
- Preferences: Edit > Preferences (on some macOS builds it is in the GIMP app menu as Settings…). Theme is under
  Interface > Theme; undo levels under System Resources.
- Saving: when the task does not ask for a file, do not save, export or overwrite: leave the edited image open
  in its window. When it asks for one, File > Export As… (⇧⌘E) with the name it gives writes PNG/JPEG; File >
  Save writes GIMP's own .xcf. Never overwrite the image that was opened (File > Overwrite, or Replace in the
  export dialog) unless the task says to replace it.
- A task that says to put or save the photo somewhere "and rename it" wants a copy under the new name: File >
  Export As… with that name and folder. Do not rename or move the opened file in Finder: the original stays where
  it was, and a check that reads it finds nothing if it is gone.
- Colors dialogs (Brightness-Contrast, Levels, Curves) are GTK: their controls are not in the controls list. Read
  them with read_screen (OCR) and use click_text, set_text_at (a number box) or drag (a slider); editing a photo
  is always possible, so do not give up with vm_fail because the list is empty. Close the dialog with OK and leave
  the edited image in front.
- The Export dialog's name field keeps the old name selected only until you click in it: select all before
  typing a new name (`set_text_at`, or ⌘A then type).
- Quitting with unsaved changes asks whether to discard them; GIMP writes its settings files when it quits.
"""

CHROME = """\
# Google Chrome

- Shown as "Google Chrome"; process "Google Chrome". It has an AppleScript dictionary (windows, tabs, each tab's
  URL and title; `execute javascript` only after View > Developer > Allow JavaScript from Apple Events is on).
- Address bar: ⌘L, type the address or search, Return. New tab ⌘T, reopen closed tab ⇧⌘T, bookmarks bar ⇧⌘B,
  bookmark this page ⌘D, history ⌘Y, print / save as PDF ⌘P.
- When Chrome was started with a debugging port, `read_app_dom('Google Chrome')` reads the page as structured text
  and `click_app_dom` clicks by label: prefer them to OCR for web pages.
- Settings pages open directly: chrome://settings/ followed by a section, e.g. privacy, security, appearance,
  fonts, languages, searchEngines, autofill, passwords, onStartup, downloads, content (site settings),
  cookies, clearBrowserData. Also chrome://history, chrome://bookmarks, chrome://downloads.
- A setting you cannot place: type a word of it into the search box at the top of chrome://settings. Some sit
  deeper than their topic suggests: "Send a Do Not Track request" is under Privacy and security > Third-party
  cookies (chrome://settings/cookies); deleting site data when all windows close is under Site settings >
  Additional content settings > On-device site data (chrome://settings/content/siteData).
- Bookmarks: ⌘D bookmarks the page; folders and edits are in the Bookmark Manager (⌥⌘B, chrome://bookmarks,
  its ⋮ menu has Add new folder / Add new bookmark). Do not script bookmark changes through AppleScript: `make
  new bookmark …` has made Chrome quit.
- Chrome keeps its settings, history and bookmarks in a profile folder that may not be the default one: change
  them in Chrome itself, never by editing or deleting its files.
- The task's page is usually already open in a tab: work there. What you leave showing in the front tab is what
  the user sees as the result.
- Finding or looking something up on the web: the answer is a page. Finish with the page that shows it open in
  the front tab, on the site the task names or the one it starts on (use that site's own search, sorting and
  filters), not with the answer only written in your reply. If that site shows an error page, go to its home
  page and navigate from there before trying another site.
- Results pages with filters (shopping, search tools, store listings) filter only through their own controls:
  click the filter's chip, checkbox or price option. Words added to the search query are not a filter. Read the
  page after each click; the filter list changes as filters apply.
- Web forms: type into a field, then wait for the page to react (suggestions, date pickers) before the next field;
  read the page again after each step that loads new content.
- Removing a site's history: open chrome://history, type the site's name in its search box, tick the first entry's
  checkbox, then shift-click the last entry's checkbox: the whole range is selected, so one Delete removes all
  that loaded. Scroll to load more and repeat. Do not delete ten entries at a time, and do not use "Clear browsing
  data" (chrome://settings/clearBrowserData) for this: it clears every site in a time range, not just the one named.
"""

THUNDERBIRD = """\
# Thunderbird

- Shown and running as "Thunderbird". No AppleScript dictionary for mail; work through its menus and dialogs.
- Settings: Thunderbird > Settings… (⌘,); account and server options: Tools > Account Settings (or Edit > Account
  Settings in some versions); message filters: Tools > Message Filters.
- Behaviour with no switch in Settings is a hidden preference: Settings > General, at the bottom, Config Editor…
  opens about:config. Search the preference by name (or by a word in it), then toggle a true/false value with its
  toggle button, or edit a value with the pencil button. Changes apply at once and are written when Thunderbird
  quits.
- Folder-specific options are in the folder's context menu > Properties….
"""

SAFARI = """\
# Safari

- Shown and running as "Safari". AppleScript dictionary: windows, tabs, URL, name; `do JavaScript` needs
  Develop > Allow JavaScript from Apple Events.
- Address bar ⌘L; new tab ⌘T; reopen closed tab ⇧⌘T; private window ⇧⌘N; settings ⌘,.
- If the task or the screen names another browser, use that one: each browser keeps its own tabs, history and
  settings.
"""

IWORK = """\
# Numbers, Pages, Keynote

- Each is shown and runs under its own name. All three have AppleScript dictionaries (documents, sheets, tables,
  cells with `value` and `formula`, slides, body text): `inspect_app_dictionary` shows the verbs.
- Read or set many cells at once with a script (`value of cell "B2" of table 1 of sheet 1 of document 1`),
  not one control at a time.
- They open .xlsx / .docx / .pptx by converting them: saving writes their own format unless you use
  File > Export To. If a task names another app (LibreOffice, Excel), do the work in that app instead.
"""

FINDER = """\
# Finder

- Shell commands through `run_osascript` (`do shell script`) are the reliable way to list, move, copy, rename
  and delete files; Finder's AppleScript dictionary also works (`move`, `duplicate`, `delete` to the Trash).
- A Finder window's title is only its folder's name. Get the full path of the folder it shows with
  `tell application "Finder" to get POSIX path of (target of front window as alias)`, then work in that
  folder, rather than searching the disk for files that are already open in front of you.
- Views: View > as Icons / as List / as Columns / as Gallery (⌘1–⌘4). Quick Look: select and press Space.
- Delete moves to the Trash; emptying the Trash is permanent.
"""

SYSTEM_SETTINGS = """\
# System Settings

- Shown as "System Settings". Its panes open directly:
  `open "x-apple.systempreferences:<pane id>"`, e.g. com.apple.Appearance-Settings.extension,
  com.apple.Sound-Settings.extension, com.apple.Desktop-Settings.extension (Desktop & Dock),
  com.apple.Keyboard-Settings.extension, com.apple.Notifications-Settings.extension,
  com.apple.Accessibility-Settings.extension, com.apple.Displays-Settings.extension,
  com.apple.ControlCenter-Settings.extension, com.apple.Users-Groups-Settings.extension.
  Ids change between macOS versions: if one does not open, use the sidebar or the search field.
- Many settings are also `defaults` keys (e.g. `defaults read com.apple.dock autohide`); a change made with
  `defaults` may need the owning process restarted (`killall Dock`) and does not show in an open pane until it
  reloads.
- Switches and pop-ups inside a pane are in the controls list; `read_dialog('System Settings')` reads a pane's
  fields with their values.
"""

NOTES = """\
# Notes

- AppleScript dictionary: folders, notes with `name` and `body` (HTML). Make or change notes with a script.
- Checklists, tables and formatting are easiest from the Format menu with the note open.
"""

PREVIEW = """\
# Preview

- Opens images and PDFs. Tools > Adjust Color… changes exposure, contrast and saturation; Tools > Adjust Size…
  scales. File > Export… writes another format. Its edits are saved to the open file.
- If the task names another image editor (GIMP), do the work there instead.
"""

QUICKTIME = """\
# QuickTime Player

- The Mac's own video and audio player: a video double-clicked in Finder opens here. Keep working in this app
  unless the task names another player; switching players loses the open document and its settings.
- View > Playback Speed lists speeds by name, not number, and the names do not say which value each is. For an
  exact speed use AppleScript: `tell application "QuickTime Player" to set rate of document 1 to <number>`
  (setting a rate above 0 also plays), and read it back with `get rate of document 1`. `inspect_app_dictionary`
  shows the document's other properties (current time, duration, playing).
"""

CARDS: dict[str, str] = {
    "libreoffice": LIBREOFFICE,
    "gimp": GIMP,
    "google chrome": CHROME,
    "thunderbird": THUNDERBIRD,
    "safari": SAFARI,
    "iwork": IWORK,
    "finder": FINDER,
    "system settings": SYSTEM_SETTINGS,
    "notes": NOTES,
    "preview": PREVIEW,
    "quicktime player": QUICKTIME,
}

ALIASES: dict[str, str] = {
    "soffice": "libreoffice", "libreoffice calc": "libreoffice", "libreoffice writer": "libreoffice",
    "libreoffice impress": "libreoffice",
    "gimp-3.2": "gimp", "gimp-3.0": "gimp", "gimp-2.10": "gimp",
    "chrome": "google chrome", "chromium": "google chrome",
    "mozilla thunderbird": "thunderbird", "thunderbird-bin": "thunderbird",
    "numbers": "iwork", "pages": "iwork", "keynote": "iwork",
    "system preferences": "system settings", "settings": "system settings",
    "quicktime": "quicktime player", "quicktimeplayerx": "quicktime player",
}


def card_key(app: str) -> str | None:
    """The card an app name (shown name, process name or a common short name) has, or None."""
    n = norm_label(app)
    if not n:
        return None
    if n in CARDS:
        return n
    if n in ALIASES:
        return ALIASES[n]
    # "GIMP 3.2", "LibreOffice Calc", "Google Chrome Canary": a card name followed by more words.
    for name in CARDS:
        if n.startswith(name + " ") or n.startswith(name + "-"):
            return name
    return None


def card_for(app: str) -> str | None:
    key = card_key(app)
    return CARDS.get(key) if key else None


def known_apps() -> list[str]:
    return sorted(CARDS)

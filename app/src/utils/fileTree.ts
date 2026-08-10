/**
 * Reconstruct a folder hierarchy from the flat file list the backend returns.
 *
 * `GET /api/sessions/{id}/files` recurses into subdirectories but only emits
 * files, so directories exist purely as "/" separators inside each path and
 * have to be inferred here.
 */

export interface SessionFileEntry {
  path: string;
  size: number;
  modified_at: number;
}

export interface FileTreeNode {
  name: string;
  path: string;
  /** Present only on leaves. */
  file?: SessionFileEntry;
  children: FileTreeNode[];
  /** Own size for a file, subtree total for a folder. */
  size: number;
  /** 1 for a file, subtree file count for a folder. */
  count: number;
}

export function buildFileTree(files: SessionFileEntry[]): FileTreeNode[] {
  const root: FileTreeNode = { name: "", path: "", children: [], size: 0, count: 0 };

  for (const file of files) {
    const parts = file.path.split("/").filter(Boolean);
    let node = root;
    parts.forEach((part, i) => {
      const isLeaf = i === parts.length - 1;
      let child = node.children.find(
        (c) => c.name === part && Boolean(c.file) === isLeaf,
      );
      if (!child) {
        child = {
          name: part,
          path: parts.slice(0, i + 1).join("/"),
          children: [],
          size: 0,
          count: 0,
        };
        if (isLeaf) child.file = file;
        node.children.push(child);
      }
      node = child;
    });
  }

  // Roll sizes and counts up from the leaves, then order folders before files.
  const settle = (node: FileTreeNode): void => {
    if (node.file) {
      node.size = node.file.size;
      node.count = 1;
      return;
    }
    node.children.forEach(settle);
    node.size = node.children.reduce((n, c) => n + c.size, 0);
    node.count = node.children.reduce((n, c) => n + c.count, 0);
    node.children.sort((a, b) => {
      if (Boolean(a.file) !== Boolean(b.file)) return a.file ? 1 : -1;
      return a.name.localeCompare(b.name);
    });
  };
  settle(root);
  return root.children;
}

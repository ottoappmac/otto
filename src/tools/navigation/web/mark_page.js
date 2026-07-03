const customCSS = `
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #27272a;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 0.375rem;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
`;

const styleTag = document.createElement("style");
styleTag.textContent = customCSS;
document.head.append(styleTag);

let labels = [];

function unmarkPage() {
  // Unmark page logic
  for (const label of labels) {
    document.body.removeChild(label);
  }
  labels = [];
}

function markPage() {
  unmarkPage();

  var bodyRect = document.body.getBoundingClientRect();

  var items = Array.prototype.slice
    .call(document.querySelectorAll("*"))
    .map(function (element) {
      var vw = Math.max(
        document.documentElement.clientWidth || 0,
        window.innerWidth || 0
      );
      var vh = Math.max(
        document.documentElement.clientHeight || 0,
        window.innerHeight || 0
      );
      var textualContent = element.textContent.trim().replace(/\s{2,}/g, " ");
      var elementType = element.tagName.toLowerCase();
      var ariaLabel = element.getAttribute("aria-label") || "";
      var src = "";
      if (element.tagName === "IFRAME") {
        src = element.getAttribute("src") || "";
      } else if (element.tagName === "A" || element.tagName === "AREA") {
        // SVG <a> exposes href as SVGAnimatedString, not a plain string
        src = (typeof element.href === "string" ? element.href : element.href.baseVal)
              || element.getAttribute("xlink:href")
              || "";
      } else if (element.getAttribute("role") === "link") {
        src = element.getAttribute("data-href")
              || element.getAttribute("data-url")
              || element.getAttribute("href")
              || "";
      } else {
        // Walk up the DOM: nearest <a> or role="link" ancestor
        let ancestor = element.parentElement;
        while (ancestor) {
          if (ancestor.tagName === "A" || ancestor.tagName === "AREA") {
            src = (typeof ancestor.href === "string" ? ancestor.href : ancestor.href.baseVal)
                  || ancestor.getAttribute("xlink:href")
                  || "";
            break;
          }
          if (ancestor.getAttribute("role") === "link") {
            src = ancestor.getAttribute("data-href")
                  || ancestor.getAttribute("data-url")
                  || "";
            break;
          }
          ancestor = ancestor.parentElement;
        }
      }

      // Final fallback: data-href/data-url on the element itself
      if (!src) {
        src = element.getAttribute("data-href") || element.getAttribute("data-url") || "";
      }

      if (src && !src.startsWith("http")) {
        // Convert relative URL to absolute URL
        const link = document.createElement("a");
        link.href = src;
        src = link.href;
      }

      var rects = [...element.getClientRects()]
        .filter((bb) => {
          var center_x = bb.left + bb.width / 2;
          var center_y = bb.top + bb.height / 2;
          var elAtCenter = document.elementFromPoint(center_x, center_y);

          return elAtCenter === element || element.contains(elAtCenter);
        })
        .map((bb) => {
          const rect = {
            left: Math.max(0, bb.left),
            top: Math.max(0, bb.top),
            right: Math.min(vw, bb.right),
            bottom: Math.min(vh, bb.bottom),
          };
          return {
            ...rect,
            width: rect.right - rect.left,
            height: rect.bottom - rect.top,
          };
        });

      var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);

      return {
        element: element,
        include:
          element.tagName === "AREA" ||
          element.tagName === "INPUT" ||
          element.tagName === "TEXTAREA" ||
          element.tagName === "SELECT" ||
          element.tagName === "BUTTON" ||
          element.tagName === "A" ||
          element.onclick != null ||
          window.getComputedStyle(element).cursor == "pointer" ||
          element.tagName === "IFRAME" ||
          element.tagName === "VIDEO" ||
          element.tagName === "TABLE" ||
          element.tagName === "LI",
        area,
        rects,
        text: textualContent,
        type: elementType,
        ariaLabel: ariaLabel,
        src: src,
      };
    })
    .filter((item) => item.include && item.area >= 20);

  // Only keep inner clickable items
  items = items.filter(
    (x) => !items.some((y) => x.element.contains(y.element) && !(x == y))
  );

  // Function to generate random colors
  function getRandomColor() {
    var letters = "0123456789ABCDEF";
    var color = "#";
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

  // Lets create a floating border on top of these elements that will always be visible
  items.forEach(function (item, index) {
    item.rects.forEach((bbox) => {
      newElement = document.createElement("div");
      var borderColor = getRandomColor();
      newElement.style.outline = `2px dashed ${borderColor}`;
      newElement.style.position = "fixed";
      newElement.style.left = bbox.left + "px";
      newElement.style.top = bbox.top + "px";
      newElement.style.width = bbox.width + "px";
      newElement.style.height = bbox.height + "px";
      newElement.style.pointerEvents = "none";
      newElement.style.boxSizing = "border-box";
      newElement.style.zIndex = 2147483647;

      // Add floating label at the top corner
      var label = document.createElement("span");
      label.textContent = index;
      label.style.position = "absolute";
      label.style.top = "0px";
      label.style.left = "0px";
      label.style.background = borderColor;
      label.style.color = "white";
      label.style.padding = "2px 4px";
      label.style.fontSize = "12px";
      label.style.borderRadius = "2px";

      // Add iframe type label
      if (item.type === "iframe") {
        const iframeLabel = document.createElement("span");
        iframeLabel.textContent = "iframe";
        iframeLabel.style.position = "absolute";
        iframeLabel.style.top = "20px";
        iframeLabel.style.left = "0px";
        iframeLabel.style.background = borderColor;
        iframeLabel.style.color = "white";
        iframeLabel.style.padding = "2px 4px";
        iframeLabel.style.fontSize = "12px";
        iframeLabel.style.borderRadius = "2px";
        newElement.appendChild(iframeLabel);
      }

      newElement.appendChild(label);

      document.body.appendChild(newElement);
      labels.push(newElement);
    });
  });

  const coordinates = items.flatMap((item, index) =>
    item.rects.map(({ left, top, width, height }) => ({
      x: (left + left + width) / 2,
      y: (top + top + height) / 2,
      type: item.type,
      text: item.text,
      ariaLabel: item.ariaLabel,
      src: item.src,
      numerical_label: index.toString()
    }))
  );
  return coordinates;
}

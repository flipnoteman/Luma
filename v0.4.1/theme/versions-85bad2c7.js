// Version switcher: lists the versions published next to this one (`/Luma/versions.json`) in
// the menu bar and links to the same page of the chosen version.
(function () {
    "use strict";

    var ROOT = "/Luma/";

    // The book is served at ROOT + "<version>/<page path>".
    var path = window.location.pathname;
    if (path.indexOf(ROOT) !== 0) {
        return; // a local preview (`mdbook serve`): nothing to switch between
    }
    var rest = path.slice(ROOT.length);
    var slash = rest.indexOf("/");
    if (slash < 0) {
        return;
    }
    var current = rest.slice(0, slash);
    var page = rest.slice(slash + 1);

    fetch(ROOT + "versions.json", { cache: "no-cache" })
        .then(function (response) {
            if (!response.ok) {
                throw new Error("versions.json: " + response.status);
            }
            return response.json();
        })
        .then(function (versions) {
            var bar = document.querySelector("#mdbook-menu-bar .left-buttons") ||
                document.querySelector("#menu-bar .left-buttons");
            if (!bar || !versions.length) {
                return;
            }
            var select = document.createElement("select");
            select.id = "luma-version-switcher";
            select.title = "Version of this guide";
            select.setAttribute("aria-label", "Version of this guide");
            versions.forEach(function (version) {
                var option = document.createElement("option");
                option.value = version;
                option.textContent = version;
                option.selected = version === current;
                select.appendChild(option);
            });
            select.addEventListener("change", function () {
                var target = ROOT + select.value + "/" + page;
                // Pages can be missing in older versions: fall back to that version's start.
                fetch(target, { method: "HEAD" })
                    .then(function (response) {
                        window.location.href = response.ok ? target : ROOT + select.value + "/";
                    })
                    .catch(function () {
                        window.location.href = ROOT + select.value + "/";
                    });
            });
            bar.appendChild(select);
        })
        .catch(function () {
            // No switcher when the index is missing.
        });
})();

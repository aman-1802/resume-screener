// Minimal progressive enhancement: auto-submit any <select data-autosubmit>
// so choosing a JD or run navigates immediately without a separate button.
document.querySelectorAll("select[data-autosubmit]").forEach(function (el) {
    el.addEventListener("change", function () {
        el.form.submit();
    });
});

// Show a lightweight "Processing..." state on multi-file screening submits,
// since scoring a batch of resumes takes a few seconds per file.
document.querySelectorAll("form").forEach(function (form) {
    form.addEventListener("submit", function () {
        var button = form.querySelector("button[type=submit]");
        if (button && form.querySelector("input[type=file][multiple]")) {
            button.disabled = true;
            button.textContent = "Processing...";
        }
    });
});

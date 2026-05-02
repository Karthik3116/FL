(function () {
    "use strict";

    const dropzone = document.getElementById("dropzone");
    if (!dropzone) return;

    const fileInput = document.getElementById("file-input");
    const dropzoneEmpty = document.getElementById("dropzone-empty");
    const dropzonePreview = document.getElementById("dropzone-preview");
    const previewImg = document.getElementById("preview-img");
    const previewName = document.getElementById("preview-name");
    const clearBtn = document.getElementById("clear-btn");
    const submitBtn = document.getElementById("submit-btn");
    const form = document.getElementById("chest-form");
    const formError = document.getElementById("form-error");

    function showError(msg) {
        if (!formError) return;
        formError.textContent = msg;
        formError.hidden = false;
    }

    function hideError() {
        if (!formError) return;
        formError.hidden = true;
        formError.textContent = "";
    }

    function setSelected(file) {
        if (!file) return;
        if (!file.type.startsWith("image/")) {
            showError("Selected file is not an image.");
            return;
        }
        hideError();
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name;
            dropzoneEmpty.hidden = true;
            dropzonePreview.hidden = false;
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
    }

    function clearSelected() {
        fileInput.value = "";
        previewImg.src = "";
        previewName.textContent = "";
        dropzoneEmpty.hidden = false;
        dropzonePreview.hidden = true;
        submitBtn.disabled = true;
        hideError();
    }

    dropzone.addEventListener("click", (e) => {
        if (e.target.closest("#dropzone-preview")) return;
        fileInput.click();
    });

    dropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach((evt) =>
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("is-dragover");
        })
    );

    ["dragleave", "drop"].forEach((evt) =>
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("is-dragover");
        })
    );

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) setSelected(files[0]);
    });

    fileInput.addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) setSelected(f);
    });

    if (clearBtn) clearBtn.addEventListener("click", clearSelected);

    if (form) {
        form.addEventListener("submit", () => {
            if (submitBtn) {
                submitBtn.classList.add("is-loading");
                submitBtn.disabled = true;
            }
        });
    }
})();

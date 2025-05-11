/*
 * @author msygrade
 * @date 2025-02-21
 * @description This script handles all the UI Interactions of Pneumonia Detection web Page.
 */

document.addEventListener("DOMContentLoaded", function () {
    console.log("Scripts loaded successfully.");

    const sections = document.querySelectorAll("section[id]");
    const navLinks = document.querySelectorAll(".contents a");

    function updateActiveLink() {
        let scrollPosition = window.scrollY + 100;

        sections.forEach((section) => {
            if (
                scrollPosition >= section.offsetTop &&
                scrollPosition < section.offsetTop + section.offsetHeight
            ) {
                navLinks.forEach((link) => link.classList.remove("active"));
                document
                    .querySelector(`.contents a[href="#${section.id}"]`)
                    .classList.add("active");
            }
        });
    }

    window.addEventListener("scroll", updateActiveLink);
});


document.addEventListener("DOMContentLoaded", function () {
    console.log("Scripts loaded successfully.");

    // ✅ Restore saved results from localStorage
    let storedFileName = localStorage.getItem("fileName");
    let storedDiagnosis = localStorage.getItem("diagnosisResult");

    if (storedFileName && storedDiagnosis) {
        document.getElementById("fileName").textContent = storedFileName;
        document.getElementById("diagnosisResult").innerHTML = `
            <p><strong>Pneumonia Detected:</strong> ${storedDiagnosis}</p>
        `;
    }
});


// ✅ Detect if page was refreshed (F5 / Ctrl+R) vs. normal navigation
window.addEventListener("load", function () {
    if (sessionStorage.getItem("reloaded")) {
        console.log("🛑 Manual Refresh Detected! Clearing stored data.");
        localStorage.removeItem("fileName");
        localStorage.removeItem("diagnosisResult");
        localStorage.removeItem("selectedFileName");
        sessionStorage.removeItem("reloaded"); // ✅ Clear flag after refresh
    }
});

// ✅ Before refresh, set session flag
window.addEventListener("beforeunload", function () {
    sessionStorage.setItem("reloaded", "true");
});

window.addEventListener("beforeunload", function () {
    console.log("💾 Clearing localStorage on app close or refresh.");
    localStorage.clear();  // ✅ Clears all localStorage items
});



let uploadedFile = null;

function uploadImage() {
    const fileInput = document.getElementById('xrayUpload');
    const file = fileInput.files[0];
    const fileNameElement = document.getElementById('fileName');
    const diagnosisResult = document.getElementById('diagnosisResult');
    const severityBar = document.getElementById('severityBar');

    if (!file) {
        alert("Please upload an X-ray image.");
        return;
    }

    diagnosisResult.innerHTML = `<p><strong>Analyzing...</strong></p>`;

    const formData = new FormData();
    formData.append("file", file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        fileNameElement.textContent = file.name;
        let diagnosisText = data.pneumonia_detected;
        let color = diagnosisText.includes("No Pneumonia") ? "#4CAF50" : "red";  // green for no pneumonia, red otherwise

        diagnosisResult.innerHTML = `
        <p><b style="color: ${color}; font-size: 22px;">${data.pneumonia_detected}</b></p>
        ${data.pneumonia_detected.includes("Pneumonia Detected") ? `<p><b style="color: red;">Severity level: ${(data.probabilities.pneumonia * 100).toFixed(2)}%</b></p>` : ''}
`;


        if (data.severity === "Normal") {
            severityBar.style.width = "0%";  // Keep empty
            severityBar.textContent = "Normal";
            severityBar.style.backgroundColor = "#4CAF50"; // Green for normal
            severityBar.style.color = "white";
        } else {
            severityBar.style.width = data.progress + "%";
            severityBar.textContent = data.severity;
            severityBar.style.backgroundColor = data.severity === "Mild" ? "#4CAF50" :
                                                data.severity === "Moderate" ? "#FFC107" : "#F44336";
        }
    })
    .catch(error => console.error('Error:', error));
}






function viewVisualization(event) {
    if (event) event.preventDefault();

    console.log("🚀 View Visualization button clicked!");

    let fileInput = document.getElementById("xrayUpload");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/visualize", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())  // ✅ Expect JSON now
    .then(data => {
        console.log("✅ Visualization Data:", data);

        if (data.success) {
            if (data.image_url) {
                window.open(data.image_url, "_blank");  // ✅ Open normal or pneumonia URL
            } else {
                // ⚡ If normal, redirect to normal_message.html
                window.open("/normal_message", "_blank");
            }
        } else {
            alert("Visualization Error: " + data.error);
        }
    })
    .catch(error => {
        console.error("❌ Fetch Visualization Error:", error);
        alert("Error fetching visualization.");
    });
}

/*
 * @author mysgrade
 * @date 2025-02-21
 * @description This script handles all the UI Interactions of Pneumonia Detection web Page.
 */

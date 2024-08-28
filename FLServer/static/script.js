// Try to retrieve the stored training type from localStorage when the script loads
let selectedTrainingType = localStorage.getItem('trainingType') || null;

function startTraining(dataset) {
    selectedTrainingType = dataset;
    localStorage.setItem('trainingType', dataset);  // Save to localStorage
    console.log("Training type set to:", selectedTrainingType);  // Log to confirm

    let endpoint = '/start_training';  // Default endpoint for training
    if (dataset.includes('medical')) {  // Check if the dataset is medical-related
        endpoint = '/start_medical_training';  // Use the medical training endpoint
    }

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ dataset: dataset })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message); // Logs the message instead of showing a popup
        // Start waiting for the training to complete
        waitForTrainingToComplete();
    });
}

// Function to wait for the server to signal a page refresh
function waitForTrainingToComplete() {
    const checkInterval = setInterval(() => {
        fetch('/should_refresh')
            .then(response => response.json())
            .then(data => {
                if (data.should_refresh) {
                    console.log("Server signaled to refresh the page.");
                    clearInterval(checkInterval);  // Stop checking after the refresh signal
                    location.reload();  // Refresh the page
                }
            })
            .catch(() => {
                console.error("Server unreachable or down.");
                clearInterval(checkInterval);  // Stop checking if the server is unreachable
            });
    },3000);  // Check every 0.01 seconds
}

function shutdownServer() {
    fetch('/shutdown', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        // Force a reload after the shutdown is initiated
        setTimeout(() => {
            location.reload();  // This will attempt to reload the page, which should fail when the server is down
        }, 2000);  // Slight delay to allow the server shutdown to complete
    })
    .catch(() => {
        // If there's an error (e.g., server is down before we reload), also try to reload
        location.reload();
    });
}

function performAdditionalRound() {
    // Retrieve from localStorage in case it's been reset
    selectedTrainingType = localStorage.getItem('trainingType');  
    console.log("Attempting to perform additional round with type:", selectedTrainingType);

    if (selectedTrainingType) {
        fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ dataset: selectedTrainingType })
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            // Start waiting for the training to complete
            waitForTrainingToComplete();
        });
    } else {
        console.error("Training type not selected.");
        alert("Training type not selected.");
    }
}

function resetServer() {
    fetch('/reset_server', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.ok) {
            console.log("Server reset successfully.");
            location.reload();  // Reload the page after reset
        } else {
            console.error("Failed to reset the server.");
        }
    })
    .catch(error => {
        console.error("Error resetting the server:", error);
    });
}

// Start listening for refresh signal immediately upon loading the page
waitForTrainingToComplete();

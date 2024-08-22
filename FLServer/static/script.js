// Try to retrieve the stored training type from localStorage when the script loads
let selectedTrainingType = localStorage.getItem('trainingType') || null;

function startTraining(dataset) {
    selectedTrainingType = dataset;
    localStorage.setItem('trainingType', dataset);  // Save to localStorage
    console.log("Training type set to:", selectedTrainingType);  // Log to confirm

    fetch('/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ dataset: dataset })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message); // Logs the message instead of showing a popup
        setTimeout(() => location.reload(), 2000); // Refresh after a delay
    });
}


function shutdownServer() {
    fetch('/shutdown', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        // Instead of showing an alert, we just reload the page
        setTimeout(() => {
            location.reload();  // This will attempt to reload the page, which will go blank because the server is down
        }, 1000);  // Slight delay to allow server shutdown to begin
    });
}

// Poll the server every 5 seconds to see if the page needs to be refreshed
setInterval(function() {
    fetch('/should_refresh')
        .then(response => response.json())
        .then(data => {
            if (data.should_refresh) {
                location.reload();
            }
        })
        .catch(() => {
            // If the server is unreachable, automatically refresh the page
            location.reload();
        });
}, 5000);

function checkServerStatus() {
    fetch('/')
        .then(response => {
            if (!response.ok) {
                throw new Error('Server not available');
            }
        })
        .catch(() => {
            // If the server is down, reload the page to show the "unreachable" state
            alert("Server is down.");
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
            setTimeout(() => location.reload(), 2000); // Refresh after a delay
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






function startTraining(dataset) {
    fetch('/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({dataset: dataset})
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message); // Logs the message instead of showing a popup
        setTimeout(() => location.reload(), 2000); // Refresh after a delay
    });
}

function refreshServer() {
    fetch('/reset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message); // Logs the message instead of showing a popup
        setTimeout(() => location.reload(), 1000); // Refresh after a delay
    });
}

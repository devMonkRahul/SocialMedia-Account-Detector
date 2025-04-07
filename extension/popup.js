document.addEventListener("DOMContentLoaded", async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
    if (tab && tab.url.includes("instagram.com")) {
      const urlParts = tab.url.split("/");
      const username = urlParts[3];
  
      if (username) {
        document.body.innerHTML += `<p>Checking <b>${username}</b>...</p>`;
  
        fetch("http://localhost:5000/analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ url: username })
        })
        .then(response => response.json())
        .then(data => {
          if (data.error) {
            document.body.innerHTML += `<p style="color: red;">Error: ${data.error}</p>`;
            return;
          }

          const result = data.isFake ? "Fake Account" : "Real Account";
          const confidence = data.isFake ? data.fakeConfidence : data.realConfidence;
          const probability = data.isFake ? data.fakeProbability : data.realProbability;

          document.body.innerHTML += `
            <div style="margin-top: 20px;">
              <h4>Prediction: ${result}</h4>
              <p>Confidence: ${confidence}</p>
              <p>Probability: ${probability}%</p>
              <img src="${data.profileImage}" style="max-width: 100px; border-radius: 50%; margin: 10px 0;">
              <p>Bio: ${data.bio}</p>
              <h5>Analysis Points:</h5>
              <ul style="list-style-type: none; padding-left: 0;">
                ${data.analysisPoints.map(point => `<li>${point}</li>`).join('')}
              </ul>
            </div>
          `;
        })
        .catch(error => {
          console.error(error);
          document.body.innerHTML += `<p style="color: red;">Error: Could not fetch prediction. Make sure the backend server is running.</p>`;
        });
      } else {
        document.body.innerHTML += "<p>No Instagram username found.</p>";
      }
    } else {
      document.body.innerHTML += "<p>Not on an Instagram profile page.</p>";
    }
  });
  
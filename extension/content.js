function getUsernameFromURL(url) {
    const pathParts = url.split('/');
    return pathParts[3] || null;
  }
  
  const username = getUsernameFromURL(window.location.href);
  
  if (username) {
    fetch(`http://localhost:5000/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url: username })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error(data.error);
        return;
      }

      const result = data.isFake ? "Fake Account" : "Real Account";
      const confidence = data.isFake ? data.fakeConfidence : data.realConfidence;
      const probability = data.isFake ? data.fakeProbability : data.realProbability;

      const box = document.createElement("div");
      box.innerHTML = `
        <div style="
          position: fixed;
          top: 20px;
          right: 20px;
          background: #222;
          color: #fff;
          padding: 15px;
          border-radius: 8px;
          z-index: 9999;
          box-shadow: 0 2px 6px rgba(0,0,0,0.3);
          max-width: 300px;
          font-family: Arial, sans-serif;
        ">
          <h3 style="margin: 0 0 10px 0; color: ${data.isFake ? '#ff4444' : '#44ff44'}">${result}</h3>
          <p style="margin: 5px 0;">Confidence: ${confidence}</p>
          <p style="margin: 5px 0;">Probability: ${probability}%</p>
          <div style="margin-top: 10px;">
            <h4 style="margin: 10px 0 5px 0;">Analysis Points:</h4>
            <ul style="margin: 0; padding-left: 10px;">
              ${data.analysisPoints.map(point => `<li style="margin: 2px 0;">${point}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
      document.body.appendChild(box);
    })
    .catch(error => {
      console.error("Error fetching prediction:", error);
    });
  }
  
document.addEventListener('DOMContentLoaded', () => {
    const profileUrl = document.getElementById('profileUrl');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const result = document.getElementById('result');
    const preview = document.getElementById('preview');
    const profileImage = document.getElementById('profileImage');
    const usernameElement = document.getElementById('username');
    const bioElement = document.getElementById('bio');
    const predictionText = document.getElementById('predictionText');
    const confidenceElement = document.getElementById('confidence');
    const analysisPoints = document.getElementById('analysisPoints');
    const predictionBanner = document.querySelector('.prediction-banner');

    // Initially hide the result
    result.classList.add('hidden');

    function validateInstagramUrl(url) {
        const instagramRegex = /^https?:\/\/(www\.)?instagram\.com\/[a-zA-Z0-9._]{1,30}\/?$/;
        return instagramRegex.test(url) || /^[a-zA-Z0-9._]{1,30}$/.test(url);
    }

    function showError(message) {
        error.textContent = message;
        error.classList.remove('hidden');
        loading.classList.add('hidden');
        result.classList.add('hidden');
        preview.classList.remove('hidden');
    }

    function resetForm() {
        profileUrl.value = '';
        error.classList.add('hidden');
        result.classList.add('hidden');
        loading.classList.add('hidden');
        preview.classList.remove('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        // Reset profile image to default
        profileImage.src = '';
        profileImage.alt = 'Profile';
    }

    function setProfileImage(imageUrl, username) {
        const imageLoader = document.getElementById('imageLoader');
        const profileImage = document.getElementById('profileImage');
        
        // Show loader and hide image
        imageLoader.classList.remove('hidden');
        profileImage.classList.add('hidden');
        
        // Create a new image object to test loading
        const img = new Image();
        
        img.onload = function() {
            // Image loaded successfully
            profileImage.src = imageUrl;
            profileImage.alt = `@${username}'s profile picture`;
            // Show image and hide loader
            profileImage.classList.remove('hidden');
            imageLoader.classList.add('hidden');
        };
        
        img.onerror = function() {
            // If image fails to load, try with a proxy first
            const proxyUrl = `https://images.weserv.nl/?url=${encodeURIComponent(imageUrl)}`;
            const proxyImg = new Image();
            
            proxyImg.onload = function() {
                profileImage.src = proxyUrl;
                profileImage.alt = `@${username}'s profile picture`;
                profileImage.classList.remove('hidden');
                imageLoader.classList.add('hidden');
            };
            
            proxyImg.onerror = function() {
                // If both direct and proxy fail, use generated avatar
                const fallbackUrl = `https://ui-avatars.com/api/?name=${username}&size=300&background=random`;
                profileImage.src = fallbackUrl;
                profileImage.alt = `Generated avatar for @${username}`;
                profileImage.classList.remove('hidden');
                imageLoader.classList.add('hidden');
            };
            
            proxyImg.src = proxyUrl;
        };
        
        // Add headers that might help with CORS and referrer issues
        img.crossOrigin = "anonymous";
        // Start loading the image
        img.src = imageUrl;
    }

    analyzeBtn.addEventListener('click', async () => {
        const url = profileUrl.value.trim();
        
        error.classList.add('hidden');
        
        if (!url) {
            showError('Please enter an Instagram profile URL or username');
            return;
        }

        if (!validateInstagramUrl(url)) {
            showError('Please enter a valid Instagram profile URL (e.g., https://www.instagram.com/username) or username');
            return;
        }

        loading.classList.remove('hidden');
        result.classList.add('hidden');
        preview.classList.add('hidden');
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');

        try {
            // Make API call to Flask backend
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url }),
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze profile');
            }
            
            loading.classList.add('hidden');
            result.classList.remove('hidden');
            preview.classList.add('hidden');
            
            // Update profile information
            usernameElement.textContent = '@' + data.username;
            setProfileImage(data.profileImage, data.username);
            bioElement.textContent = data.bio || 'No bio available';
            
            // Update result status with both real and fake probabilities
            const status = document.querySelector('.status');
            
            if (data.isFake) {
                status.className = 'status rounded-lg p-4 text-center bg-gradient-to-r from-red-500 to-red-600 text-white';
                status.innerHTML = `
                    <span class="status-text text-lg font-bold tracking-wide">FAKE PROFILE</span>
                    <div class="mt-3 space-y-2">
                        <div class="flex justify-between items-center px-3 py-1 rounded-full bg-white/20">
                            <span>Fake Score:</span>
                            <span class="font-bold">${data.fakeProbability}% (${data.fakeConfidence})</span>
                        </div>
                        <div class="flex justify-between items-center px-3 py-1 rounded-full bg-white/20">
                            <span>Real Score:</span>
                            <span class="font-bold">${data.realProbability}% (${data.realConfidence})</span>
                        </div>
                    </div>
                `;
            } else {
                status.className = 'status rounded-lg p-4 text-center bg-gradient-to-r from-green-500 to-green-600 text-white';
                status.innerHTML = `
                    <span class="status-text text-lg font-bold tracking-wide">GENUINE PROFILE</span>
                    <div class="mt-3 space-y-2">
                        <div class="flex justify-between items-center px-3 py-1 rounded-full bg-white/20">
                            <span>Real Score:</span>
                            <span class="font-bold">${data.realProbability}% (${data.realConfidence})</span>
                        </div>
                        <div class="flex justify-between items-center px-3 py-1 rounded-full bg-white/20">
                            <span>Fake Score:</span>
                            <span class="font-bold">${data.fakeProbability}% (${data.fakeConfidence})</span>
                        </div>
                    </div>
                `;
            }
            
            // Update confidence level
            if (confidenceElement) {
                confidenceElement.textContent = `Confidence Level: ${data.confidence}`;
            }
            
            // Update analysis points
            if (analysisPoints) {
                analysisPoints.innerHTML = '';
                data.analysisPoints.forEach(point => {
                    const li = document.createElement('li');
                    li.textContent = point;
                    analysisPoints.appendChild(li);
                });
            }

            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        } catch (error) {
            showError(error.message);
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    });

    resetBtn.addEventListener('click', resetForm);

    profileUrl.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeBtn.click();
        }
    });
});


//The interface now expects your backend to return data in this format:
// {
//     "profileImage": "URL to the profile image",
//     "username": "Profile username",
//     "bio": "Profile bio",
//     "isFake": true/false,
//     "confidence": 95,
//     "analysisPoints": [
//         "Point 1 about the analysis",
//         "Point 2 about the analysis",
//         "Point 3 about the analysis"
//     ]
// }
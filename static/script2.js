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

    // Model comparison elements
    const tfRealProb = document.getElementById('tfRealProb');
    const tfFakeProb = document.getElementById('tfFakeProb');
    const tfRealBar = document.getElementById('tfRealBar');
    const tfFakeBar = document.getElementById('tfFakeBar');
    const xgbRealProb = document.getElementById('xgbRealProb');
    const xgbFakeProb = document.getElementById('xgbFakeProb');
    const xgbRealBar = document.getElementById('xgbRealBar');
    const xgbFakeBar = document.getElementById('xgbFakeBar');
    const ensembleRealProb = document.getElementById('ensembleRealProb');
    const ensembleFakeProb = document.getElementById('ensembleFakeProb');
    const ensembleRealBar = document.getElementById('ensembleRealBar');
    const ensembleFakeBar = document.getElementById('ensembleFakeBar');
    const finalVerdict = document.getElementById('finalVerdict');
    const confidenceLevel = document.getElementById('confidenceLevel');

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
        profileImage.src = '';
        profileImage.alt = 'Profile';
    }

    function setProfileImage(imageUrl, username) {
        const imageLoader = document.getElementById('imageLoader');
        
        // Show loader and hide image
        imageLoader.classList.remove('hidden');
        profileImage.classList.add('hidden');
        
        // Create a new image object to test loading
        const img = new Image();
        
        img.onload = function() {
            profileImage.src = imageUrl;
            profileImage.alt = `@${username}'s profile picture`;
            profileImage.classList.remove('hidden');
            imageLoader.classList.add('hidden');
        };
        
        img.onerror = function() {
            const proxyUrl = `https://images.weserv.nl/?url=${encodeURIComponent(imageUrl)}`;
            const proxyImg = new Image();
            
            proxyImg.onload = function() {
                profileImage.src = proxyUrl;
                profileImage.alt = `@${username}'s profile picture`;
                profileImage.classList.remove('hidden');
                imageLoader.classList.add('hidden');
            };
            
            proxyImg.onerror = function() {
                const fallbackUrl = `https://ui-avatars.com/api/?name=${username}&size=300&background=random`;
                profileImage.src = fallbackUrl;
                profileImage.alt = `Generated avatar for @${username}`;
                profileImage.classList.remove('hidden');
                imageLoader.classList.add('hidden');
            };
            
            proxyImg.src = proxyUrl;
        };
        
        img.crossOrigin = "anonymous";
        img.src = imageUrl;
    }

    function updateProgressBar(bar, value) {
        bar.style.width = `${value}%`;
    }

    function updateModelComparison(data) {
        // Update TensorFlow model results
        const tfReal = data.modelComparison.tensorflow.realProbability;
        const tfFake = data.modelComparison.tensorflow.fakeProbability;
        tfRealProb.textContent = `${tfReal.toFixed(1)}%`;
        tfFakeProb.textContent = `${tfFake.toFixed(1)}%`;
        updateProgressBar(tfRealBar, tfReal);
        updateProgressBar(tfFakeBar, tfFake);

        // Update XGBoost model results
        const xgbReal = data.modelComparison.xgboost.realProbability;
        const xgbFake = data.modelComparison.xgboost.fakeProbability;
        xgbRealProb.textContent = `${xgbReal.toFixed(1)}%`;
        xgbFakeProb.textContent = `${xgbFake.toFixed(1)}%`;
        updateProgressBar(xgbRealBar, xgbReal);
        updateProgressBar(xgbFakeBar, xgbFake);

        // Update Ensemble results
        const ensembleReal = data.realProbability;
        const ensembleFake = data.fakeProbability;
        ensembleRealProb.textContent = `${ensembleReal.toFixed(1)}%`;
        ensembleFakeProb.textContent = `${ensembleFake.toFixed(1)}%`;
        updateProgressBar(ensembleRealBar, ensembleReal);
        updateProgressBar(ensembleFakeBar, ensembleFake);

        // Update final verdict
        const isFake = data.isFake;
        finalVerdict.textContent = isFake ? 'FAKE PROFILE' : 'REAL PROFILE';
        finalVerdict.className = `text-xl font-bold ${isFake ? 'text-red-600' : 'text-green-600'}`;
        
        // Update confidence level
        const confidence = isFake ? data.fakeConfidence : data.realConfidence;
        confidenceLevel.textContent = `Confidence Level: ${confidence}`;
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
            
            // Update model comparison
            updateModelComparison(data);

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
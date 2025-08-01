// Debug script to clear cache and test API
console.log('🔄 Force clearing API cache...');
window.apiService.clearCache();

console.log('🌐 Testing backend connection...');
window.apiService.checkBackendHealth().then(healthy => {
    console.log('Backend healthy:', healthy);
    
    console.log('📊 Force fetching algorithms...');
    return window.apiService.forceRefreshAlgorithms();
}).then(data => {
    console.log('✅ Algorithms fetched successfully:', data);
    console.log('Cache cleared and fresh data loaded!');
}).catch(error => {
    console.error('❌ Error:', error);
});

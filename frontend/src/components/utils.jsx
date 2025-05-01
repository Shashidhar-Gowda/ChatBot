// utils/authFetch.js
export const authFetch = async (url, options = {}) => {
    let token = document.cookie.split('; ').find(row => row.startsWith('token='))?.split('=')[1]
        || localStorage.getItem('token');
    let refreshToken = localStorage.getItem('refresh');

    if (!options.headers) {
        options.headers = {};
    }

    options.headers['Authorization'] = `Bearer ${token}`;

    let response = await fetch(url, options);

    if (response.status === 401 && refreshToken) {
        // Access token might be expired — try refreshing
        const refreshRes = await fetch('http://127.0.0.1:8000/api/token/refresh/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh: refreshToken }),
        });

        if (refreshRes.ok) {
            const data = await refreshRes.json();
            const newAccess = data.access;

            // Store new token
            document.cookie = `token=${newAccess}; path=/`;
            localStorage.setItem('token', newAccess);

            // Retry original request with new token
            options.headers['Authorization'] = `Bearer ${newAccess}`;
            return fetch(url, options);
        } else {
            // Refresh token invalid — force logout or redirect
            window.location.href = '/login';
            throw new Error("Session expired. Please login again.");
        }
    }

    return response;
};

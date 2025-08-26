<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BELUGA Dashboard 🐋🐪</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios@1.4.0/dist/axios.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.9/babel.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-black text-green-400">
    <div id="root"></div>
    <script type="text/babel">
        const Dashboard = () => {
            const [data, setData] = React.useState({
                threat_data: {},
                visualization_data: {},
                navigation_data: [],
                optimization_data: {}
            });
            const [error, setError] = React.useState(null);
            const [environment, setEnvironment] = React.useState('space');

            const fetchDashboardData = async () => {
                try {
                    const response = await axios.post('http://localhost:8000/api/services/beluga_dashboard', {
                        environment,
                        oauth_token: localStorage.getItem('oauth_token') || 'your-oauth-token',
                        security_mode: 'advanced',
                        wallet_address: 'f2a3b4c5-d6e7-4f8a-90ab-e1c2d3e4f5a6',
                        reputation: 2500000000
                    });
                    setData(response.data);
                    setError(null);
                } catch (err) {
                    setError('Failed to fetch dashboard data: ' + err.message);
                }
            };

            React.useEffect(() => {
                fetchDashboardData();
            }, [environment]);

            return (
                <div className="container mx-auto p-4">
                    <h1 className="text-3xl font-bold mb-4">BELUGA Dashboard 🐋🐪</h1>
                    <select
                        className="bg-gray-800 text-green-400 p-2 rounded mb-4"
                        value={environment}
                        onChange={(e) => setEnvironment(e.target.value)}
                    >
                        <option value="space">Space</option>
                        <option value="arctic">Arctic</option>
                        <option value="desert">Desert</option>
                        <option value="submarine">Submarine</option>
                        <option value="subterranean">Subterranean</option>
                    </select>
                    {error && <p className="text-red-500">{error}</p>}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <h2 className="text-xl font-semibold">Threat Data</h2>
                            <pre className="bg-gray-800 p-4 rounded">{JSON.stringify(data.threat_data, null, 2)}</pre>
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold">Visualization Data</h2>
                            <pre className="bg-gray-800 p-4 rounded">{JSON.stringify(data.visualization_data, null, 2)}</pre>
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold">Navigation Data</h2>
                            <pre className="bg-gray-800 p-4 rounded">{JSON.stringify(data.navigation_data, null, 2)}</pre>
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold">Optimization Data</h2>
                            <pre className="bg-gray-800 p-4 rounded">{JSON.stringify(data.optimization_data, null, 2)}</pre>
                        </div>
                    </div>
                </div>
            );
        };

        ReactDOM.render(<Dashboard />, document.getElementById('root'));
    </script>
</body>
</html>

<!-- Deployment Instructions -->
<!-- Path: webxos-vial-mcp/src/frontend/beluga_react_dashboard.jsx -->
<!-- Run: Serve via a static file server (e.g., nginx) or integrate with a Node.js server -->
<!-- Example: docker run -p 80:80 -v $(pwd)/src/frontend:/usr/share/nginx/html nginx -->

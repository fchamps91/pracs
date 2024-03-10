const http = require('http');
const hostname = '0.0.0.0'; 
const port = 3000;
const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/plain');
  res.end('Hello world!\n Abdulkader Kanchwala 033');
});
server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});


//docker build -t aknodejs-app .
//docker run -p 3000:3000 name
const express = require('express');
const app = express();
const port = 3000;

function checkOddEven(number) {
  return number % 2 === 0 ? "Even" : "Odd";
}

// route to handle requests
app.get('/:number', (req, res) => {
  const numberToCheck = parseInt(req.params.number, 10);
  const result = checkOddEven(numberToCheck);
  res.send(`${numberToCheck} is ${result}.`);
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});


//docker build -t aknodejs-app .
//docker run -p 3000:3000 name
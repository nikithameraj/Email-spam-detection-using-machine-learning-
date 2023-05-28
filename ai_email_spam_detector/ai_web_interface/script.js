const form = document.getElementById('spamForm');
const resultContainer = document.getElementById('resultContainer');

form.addEventListener('submit', function(event) {
  event.preventDefault();
  const message = document.querySelector('textarea[name="message"]').value;

  const url = `http://127.0.0.1:8000/predict/?message=${encodeURIComponent(message)}`;

  fetch(url, {
    method: 'POST',
    headers: {
      'Accept': 'application/json'
    },
  })
  .then(response => response.json())
  .then(result => {
    // Handle the response
    console.log(result);

    if (result) {
      resultContainer.innerHTML = `
        <div class="alert alert-danger" role="alert">
          <strong>Spam:</strong> This message is flagged as spam.
        </div>
      `;
    } else {
      resultContainer.innerHTML = `
        <div class="alert alert-success" role="alert">
          <strong>Not Spam:</strong> This message is not flagged as spam.
        </div>
      `;
    }
  })
  .catch(error => {
    // Handle any errors
    console.error('Error:', error);
  });
});

// NavBar scroll watcher
const navBar = document.querySelector('.navBar');       // Select the NavBar element
const scrollWatcher = document.createElement('div');    // Create a div to act as a scroll watcher

scrollWatcher.setAttribute('data-scroll-watcher', '');  // Add an attribute for styling
navBar.before(scrollWatcher);                           // Insert the scroll watcher before the NavBar

// Set up an IntersectionObserver to monitor the scroll watcher
const navObserver = new IntersectionObserver((entries) => {
    navBar.classList.toggle('sticking', !entries[0].isIntersecting); // Note: inverted logic for sticky behavior
}, {
    threshold: [1], // Trigger when the scroll watcher is fully in view
    rootMargin: '150px 0px 0px 0px' // Adds a 150px top margin to the root
}); 

navObserver.observe(scrollWatcher); // Start observing the scroll watcher
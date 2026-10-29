// Copy BibTeX to clipboard
function copyBibTeX() {
    const text = document.getElementById('bibtex-code').textContent;
    const button = document.querySelector('.copy-bibtex-btn');
    const label = button.querySelector('.copy-text');
    const done = function () {
        button.classList.add('copied');
        label.textContent = 'Cop';
        setTimeout(function () {
            button.classList.remove('copied');
            label.textContent = 'Copy';
        }, 2000);
    };
    navigator.clipboard.writeText(text).then(done).catch(function () {
        const area = document.createElement('textarea');
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand('copy');
        document.body.removeChild(area);
        done();
    });
}

// Scroll to top
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
    document.querySelector('.scroll-to-top').classList.toggle('visible', window.pageYOffset > 300);
});

(function() {
    const buttons = document.getElementsByClassName("theme-toggle");
    Array.from(buttons).forEach((btn) => {
        btn.addEventListener("click", (e) => { 
            const currentTheme = localStorage.getItem("theme");
            const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
            if (currentTheme === (prefersDark ? "dark" : "light")) {
                // Skip the "auto" theme
                document.querySelector('.theme-toggle').click();
            }
        });
    });
})();
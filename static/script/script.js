$(document).ready(function () {
    $(".dashboard-nav-dropdown-toggle").click(function () {
        $(this).closest(".dashboard-nav-dropdown")
            .toggleClass("show")
            .find(".dashboard-nav-dropdown")
            .removeClass("show");
        $(this).parent()
            .siblings()
            .removeClass("show");
    });
    $(".menu-toggle").click(function () {
        if (mobileScreen.matches) {
            $(".dashboard-nav").toggleClass("mobile-show");
        } else {
            $(".dashboard").toggleClass("dashboard-compact");
        }
    });
    $('#uploadForm').on('submit', function(event) {
        event.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: "/upload_video",
            type: "POST",
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            success: function(response) {
                alert(response);
            },
            error: function(jqXHR, textStatus, errorMessage) {
                console.log(errorMessage);
            }
        });
    });
});

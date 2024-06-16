const mobileScreen = window.matchMedia("(max-width: 990px )");
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
                console.log("Upload success");
            },
            error: function(jqXHR, textStatus, errorMessage) {
                console.log(errorMessage);
            }
        });
    });
    $('#yolo-version').change(function(){
        var selectedVersion = $(this).val();
        $.ajax({
            url: '/yolo_version',
            type: 'post',
            data: { 'yolo-version': selectedVersion },
            success: function(response) {
                console.log("Yolo version changed to " + selectedVersion);
                // Update the src attribute of the img tag to include the new yolo-version
                $('img').attr('src', '/video_feed?yolo-version=' + selectedVersion);
                
            }
        });
    });
    $('#yolo-version').change(function() {
        $.ajax({
            url: '/set_pass_liveness',  // Replace with your actual route
            type: 'POST',
            data: { pass_liveness: false },
            success: function(response) {
                $('#face-reg').prop('disabled', true);
            }
        });
    });
    function checkLiveness() {
        $.getJSON('/get_pass_liveness', function(data) {
            if (data.pass_liveness) {
                $('#face-reg').prop('disabled', false);
            }
        });
    }
    setInterval(checkLiveness, 1000);  // Check every second

    function fetchVideoFeed() {
        const videoFeed = $('#video-feed');
        videoFeed.attr('src', "{{ url_for('video_feed') }}");
    }
    fetchVideoFeed();
});




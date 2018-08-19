function dispLoading(msg) {
    console.log("dipLoading was called!");
    if (msg == undefined) {
        msg = "";
    }
    var dispMsg = "<div class='loadingMsg'>" + msg + "</div>";
    if ($("#loading").length == 0) {
        $("body").append("<div id='loading'>" + dispMsg + "</div>");
    }
}

function removeLoading() {
    $("#loading").remove();
}
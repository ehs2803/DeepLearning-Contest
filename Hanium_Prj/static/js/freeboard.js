/* 게시판 임시 */
$(".w3-pagination a").click(function() {

  var selector = $(this);

  var current_index = $(".w3-pagination a").index(this);
  //alert(current_index);
  var current_class = selector.attr("class");

  if (current_index != 0 || current_index != 6) {

    $(".w3-pagination a").each(function(index) {

      if (index == 0 || index == 6) {
        return true;
      } else {
        if (index == current_index) {
          $(this).addClass("w3-green");
        } else {
          $(this).removeClass("w3-green");
        }
      }

    });

  }

});
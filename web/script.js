/*var data = [[10, 14, 22, 17, 21, 22, 16, 11, -1, 17, 10, 15, 8, 26, 22],
[2, -1, 20, -1, 3, -1, 3, -1, 3, -1, 18, -1, 3, -1, 13],
[15, 2, 13, 22, 20, -1, 7, 5, 3, 26, 17, -1, 25, 18, 18],
[24, -1, 3, -1, 11, -1, 17, -1, 20, -1, 7, -1, 25, -1, 3],
[10, 3, 14, 11, -1, 1, 24, 3, 26, 11, -1, 18, 22, 20, 7],
[-1, -1, 22, -1, 25, -1, -1, -1, 4, -1, -1, -1, 3, -1, 18],
[10, 16, 2, 1, 22, 11, 7, 20, -1, 1, 3, 7, 18, 7, -1],
[15, -1, -1, -1, 3, -1, 22, -1, 18, -1, 20, -1, -1, -1, 7],
[3, 22, 6, 6, 25, -1, 24, 16, 7, 3, 22, 17, 25, 1, 24],
[12, -1, 25, -1, 25, -1, 22, -1, 6, -1, 19, -1, 18, -1, 18],
[7, 6, 24, 3, 7, 12, 26, 1, 24, -1, 2, 22, 4, 18, 20],
[3, -1, -1, -1, 1, -1, 14, -1, 15, -1, 22, -1, -1, -1, 20],
[-1, 25, 19, 25, 1, 11, -1, 22, 2, 24, 25, 1, 24, 25, 18],
[1, -1, 21, -1, -1, -1, 14, -1, -1, -1, 13, -1, 19, -1, -1],
[16, 3, 14, 24, -1, 20, 15, 25, 1, 4, -1, 25, 3, 25, 1],
[15, -1, 19, -1, 16, -1, 24, -1, 22, -1, 24, -1, 18, -1, 14],
[24, 15, 18, -1, 7, 5, 3, 22, 14, -1, 7, 8, 3, 18, 7],
[16, -1, 18, -1, 23, -1, 1, -1, 9, -1, 20, -1, 18, -1, 7],
[19, 25, 11, 26, 20, 13, -1, 25, 15, 15, 13, 16, 16, 25, 17]];*/

function CreateGrid(data)
{
    // data is an array of arrays
    var width = data[0].length;
    var height = data.length;
    var table = document.getElementById('grid');

    for (var r = 0; r < data.length; r++) {
        var row = data[r];
        var tr = document.createElement('tr');

        for (var c = 0; c < row.length; c++) {
            var td = document.createElement('td');
            var number = row[c];

            if (number === -1) {
                td.className = 'blackbox';
            } else {
                var numberText = document.createElement('span');
                numberText.innerText = row[c];
                numberText.className = 'number';

                var text = document.createElement('span');
                text.innerText = 'A';
                text.className = 'letter';

                td.appendChild(numberText);
                td.appendChild(text);
            }

            tr.appendChild(td);
        }

        table.appendChild(tr);
    }
}

function ProcessImage()
{
    $.get("/image?filename=" + $("#filename").val(), function(data) {
        $("#crossword_data").val(data);
    });
}

function ProcessStructure()
{
    $.ajax("/solve", {
        data: $("#crossword_data").val(),
        contentType: 'application/json',
        type : 'POST'

    });
    $.get("/image?filename=" + $("#filename").val(), function(data) {
        $("#crossword_data").val(data);
    });
}

$(function() {
    $('#do_filename').click(function() {
        ProcessImage();
        return false;
    });

    $('#do_structure').click(function() {
        CreateGrid(JSON.parse($("#crossword_data").val()));
        ProcessStructure();
        return false;
    });
});

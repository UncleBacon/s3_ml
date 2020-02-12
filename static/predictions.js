// Hide the button to add a new image until the first result is rendered
document.getElementsByClassName('show')[1].style.visibility = 'hidden';

// Control the progress in the bar in the html code
$(function () {
    var current_progress = 0;
    var interval = setInterval(function () {
        current_progress += 10;
        $("#dynamic")
            .css("width", current_progress + "%")
            .attr("aria-valuenow", current_progress)
            .text(current_progress + "% Complete");
        if (current_progress >= 90)
            clearInterval(interval);
    }, 1000);
});

// Flask Route that contains the predicted data from the models
var url = `/predict`;

// Variable initialization for the information from flask for the predictions
// and probabilities
var model = [],
    no = [],
    prediction = [],
    probability = [];

// Create arrays for all the data that comes from flask in JSON format
d3.json(url).then((data) => {
    // console.log(data)
    for (i = 0; i < 3; i++) {
        data[i].forEach((item) => {
            model.push(item.Model)
            no.push(item.no)
            prediction.push(item.prediction)
            probability.push(item.probability)
        })
    }

    // Check the array of information in console
    // console.log(model),
    //     console.log(no),
    //     console.log(prediction),
    //     console.log(probability);

    // Google library for Gauges 
    google.charts.load('current', {
        callback: function () {
            var container = document.getElementById('chart_div');
            var chart = new google.visualization.Gauge(container);

            // Convert the probabilities to a float number
            var p1 = parseFloat(probability[0])
            var p2 = parseFloat(probability[3])
            var p3 = parseFloat(probability[6])

            // Verify values of probabilities in console
            // console.log(p1)
            // console.log(p2)
            // console.log(p3)

            // Input the information to the Gauge
            var data = google.visualization.arrayToDataTable([
                ['Label', 'Value'],
                ['VGG16', p1],
                ['VGG19', p2],
                ['RESNET50', p3]
            ]);

            // Create percentage format for the probabilities
            var formatter = new google.visualization.NumberFormat(
                { suffix: '%', pattern: '#' }
            );
            formatter.format(data, 1);
            
            // Options for the Google Gauge Library
            var options = {
                min: 0, max: 100,
                width: 500, height: 255,
                redFrom: 0, redTo: 25,
                yellowFrom: 25, yellowTo: 75,
                greenFrom: 75, greenTo: 100,
                minorTicks: 10
            };

            // Render the Gauge
            chart.draw(data, options);
        },
        packages: ['gauge']
    });

    // Load the Google Library for Charts
    google.charts.load('current', { 'packages': ['table'] });
    google.charts.setOnLoadCallback(drawTable);


    function drawTable() {
        // Set the columns of the chart
        var data = new google.visualization.DataTable();
        data.addColumn('number', 'Probability');
        data.addColumn('string', 'Prediction')
        data.addColumn('string', 'Model');
        data.addColumn('number', 'Model Prediction');
        ;

        // Add the columns to the table from the Flask information in JSON format
        for (i = 0; i < 9; i++) {
            data.addRows([
                [{ v: parseFloat(probability[i]), f: probability[i] }, prediction[i][0], model[i], no[i]]
            ])
        }


        // sort by column 0, descending
        data.sort({ column: 0, desc: true });

        // Create the table
        var table = new google.visualization.Table(document.getElementById('table_div'));
        table.draw(data, { showRowNumber: true, width: '100%', height: '100%' });

        // Hide the progress bar once the Gauge and Table are created
        document.getElementsByClassName('hide')[0].style.visibility = 'hidden';
        // Show the title for the Gauge and Table once the Gauge and Table are created
        document.getElementsByClassName('show')[0].style.visibility = 'visible';
        // Show the button for a new image once the Gauge and Table are created
        document.getElementsByClassName('show')[1].style.visibility = 'visible';
        // Create the title for the Gauge and Table once the Gauge and Table are created
        document.getElementsByClassName('show')[0].innerHTML = 'Predictions';

        // Render the information down the Google Gauges that cannot be rendered from the library
        document.getElementsByClassName('label')[0].innerHTML = prediction[0][0];
        document.getElementsByClassName('label')[1].innerHTML = prediction[3][0];
        document.getElementsByClassName('label')[2].innerHTML = prediction[6][0];
    }
})


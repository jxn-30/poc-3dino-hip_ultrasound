Loss and Dice were extracted from training.log via simple node commands, then turned into charts with https://www.chartjs.org

JSON.stringify(fs.readFileSync('training.log').toString().match(/(?<=Iter \d+], Train loss: )([\d.]+)(?=$)/gm).map(loss => parseFloat(loss)))

JSON.stringify(fs.readFileSync('training.log').toString().match(/(?<=Val dice: )([\d.]+)(?=$)/gm).map(dice => parseFloat(dice)))

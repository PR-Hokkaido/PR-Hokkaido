(function() {
    var loading = document.getElementById('loading');
    var hiddenArea = document.getElementsByClassName('load-on-hidden')
    var resultField = document.getElementById('result');
    var resultImgField = document.getElementById('resultImg');
    var resultKarasuField = document.getElementById('result_karasu');
    var resultSuzumeField = document.getElementById('result_suzume');
    var resultTsuruField = document.getElementById('result_tsuru');
    var resultShimaenagaField = document.getElementById('result_shimaenaga');
    var birdName = {karasu:"カラス", suzume:"スズメ", tsuru:"ツル", shimaenaga:"シマエナガ"};

    window.addEventListener('load', function() {
        // 画像判定実行
        var result = execAIasc();
    });

    async function execAIasc() {
        // 判定実行
        var val = await eel.execAIasc()();
        var JSONval = JSON.parse(val);
        console.log(JSONval);

        // 判定画像を設定
        var imgTag = document.createElement('img');
        imgTag.src = "data/" + JSONval[0]["img"];
        resultImgField.appendChild(imgTag);

        // 判定結果を設定
        var result = judge(JSONval);
        resultField.innerHTML = birdName[result];

        // 判定結果詳細を設定
        /*
        resultKarasuField.innerHTML = Math.round(JSONval[0]["karasu"]*1000)/10;
        resultSuzumeField.innerHTML = Math.round(JSONval[0]["suzume"]*1000)/10;
        resultTsuruField.innerHTML = Math.round(JSONval[0]["tsuru"]*1000)/10;
        resultShimaenagaField.innerHTML = Math.round(JSONval[0]["shimaenaga"]*1000)/10;
        */
        resultKarasuField.innerHTML = (JSONval[0]["karasu"]*100).toFixed(1);
        resultSuzumeField.innerHTML = (JSONval[0]["suzume"]*100).toFixed(1);
        resultTsuruField.innerHTML = (JSONval[0]["tsuru"]*100).toFixed(1);
        resultShimaenagaField.innerHTML = (JSONval[0]["shimaenaga"]*100).toFixed(1);

        // ローディング画面解除
        hideLoading();

        return JSONval;
    }

    function judge(data) {
        var result = "";
        var maxKakuritu = 0;
        for (key in data[0]) {
            if ((typeof data[0][key]) == "number" && maxKakuritu < data[0][key]){
                result = key;
                maxKakuritu = data[0][key]
            }
        }
        return result;
    }

    function hideLoading() {
        // ドロップ領域の見た目を変更
        loading.classList.remove('load-on');
        loading.classList.add('load-off');

        for(var i = 0; i < hiddenArea.length; i++) {
            hiddenArea[i].style.display = "flex";
        }
    }
})();

$(function() {
    //videoタグを取得
    var video = document.getElementById('camera');
    //カメラが起動できたかのフラグ
    var localMediaStream = null;
    //カメラ使えるかチェック
    var hasGetUserMedia = function() {
        return (navigator.getUserMedia || navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia || navigator.msGetUserMedia);
    };

    //エラー
    var onFailSoHard = function(e) {
        console.log('エラー!', e);
    };

    if(!hasGetUserMedia()) {
        alert("未対応ブラウザです。");
    } else {
        window.URL = window.URL || window.webkitURL;
        navigator.getUserMedia  = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
        navigator.getUserMedia({video: true}, function(stream) {
            //video.src = window.URL.createObjectURL(stream);
            video.srcObject = stream;
            localMediaStream = stream;
        }, onFailSoHard);
    }

    $("#judgeBtn").click(function() {
        if (localMediaStream) {
            var canvas = document.getElementById('canvas');
            //canvasの描画モードを2sに
            var ctx = canvas.getContext('2d');
            var img = document.getElementById('img');

            //videoの縦幅横幅を取得
            var w = video.offsetWidth;
            var h = video.offsetHeight;

            //同じサイズをcanvasに指定
            canvas.setAttribute("width", w);
            canvas.setAttribute("height", h);

            //canvasにコピー
            ctx.drawImage(video, 0, 0, w, h);
            //imgにjpg形式で書き出し
            img.src = canvas.toDataURL('image/jpg');

            // 画像フォルダを初期化
            imgDelete();

            // 読み込んだファイルを画像化
            var src = img.src;
            // データ部だけ切り出し
            var data = src.split( "," )[1];
            imgCreate(0,data);

            // 結果画面へ遷移（判定は結果画面で実施）
            location.href = 'result.html';
		}
	});


});

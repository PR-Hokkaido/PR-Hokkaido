(function() {
    var URL_BLANK_IMAGE = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    var elDrop = document.getElementById('dropzone');
    var elFiles = document.getElementById('files');
    var judgeBtn = document.getElementById('judgeBtn');

    judgeBtn.addEventListener('click', function(event) {
        var test = document.getElementById('files');
        if(test.children.length > 0) {
            // 画像フォルダを初期化
            imgDelete();

            // 読み込んだファイルを画像化
            for (var i = 0; i < test.children.length; i++) {
                var src = test.children[i].src;
                // データ部だけ切り出し
                var data = src.split( "," )[1];
                imgCreate(i,data);
            }

            // 結果画面へ遷移（判定は結果画面で実施）
            location.href = 'result.html';
        }
    });

    elDrop.addEventListener('dragover', function(event) {
        // イベントキャンセル
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
        // ドロップ領域の見た目を変更
        showDropping();
    });

    elDrop.addEventListener('dragleave', function(event) {
        // ドロップ領域の見た目を変更
        hideDropping();
    });

    elDrop.addEventListener('drop', function(event) {
        // イベントキャンセル
        event.preventDefault();
        // ドロップ領域の見た目を変更
        hideDropping();

        // ドロップされたファイルを取得
        var files = event.dataTransfer.files;
        showFiles(files);
    });

    document.addEventListener('click', function(event) {
        var elTarget = event.target;
        if (elTarget.tagName === 'IMG') {
            var src = elTarget.src;
            var w = window.open('about:blank');
            var d = w.document;
            var title = escapeHtml(elTarget.getAttribute('title'));

            d.open();
            d.write('<title>' + title + '</title>');
            d.write('<img src="' + src + '" />');
            d.close();
        }
    });

    function showDropping() {
        // ドロップ領域の見た目を変更
        elDrop.classList.add('dropover');
    }

    function hideDropping() {
        // ドロップ領域の見た目を変更
        elDrop.classList.remove('dropover');
    }

    function showFiles(files) {
        elFiles.innerHTML = '';

        // ファイル数分ループ
        for (var i=0, l=files.length; i<l; i++) {
            var file = files[i];
            var elFile = buildElFile(file);
            elFiles.appendChild(elFile);
        }
    }

    function buildElFile(file) {
        // ファイルタイプが画像の場合
        if (file.type.indexOf('image/') === 0) {
            // IMGタグを作成
            var elImage = document.createElement('img');
            elImage.src = URL_BLANK_IMAGE;
            attachImage(file, elImage);
            return elImage;
        }
    }

    function attachImage(file, elImage) {
        // file：ファイル本体情報
        // elImage：作成したIMGタグ
        var reader = new FileReader();
        // readAsDataURLで画像読み込みが成功した場合に発火するイベント
        reader.onload = function(event) {
            // 読み込んだ画像のDataURIがevent.target.resultに入っている
            var src = event.target.result;
            elImage.src = src;
            elImage.setAttribute('title', file.name);
        };
        reader.readAsDataURL(file);
    }

    function escapeHtml(source) {
        var el = document.createElement('div');
        el.appendChild(document.createTextNode(source));
        var destination = el.innerHTML;
        return destination;
    }
})();

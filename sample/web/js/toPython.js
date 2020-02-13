// 画像フォルダを初期化
async function imgDelete() {
    await eel.imgDelete();
    return true;
}

// 読み込んだ画像をファイル可
async function imgCreate(i, data) {
    await eel.imgCreate(i,data);
    return true;
}

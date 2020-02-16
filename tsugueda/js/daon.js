$(function() {
    $('#btn_start_recording').on('click', function(){
      startRecording();
    })
  
    $('#btn_stop_recording').on('click', function(){
      endRecording();
    })
    
    $('#btn_classification').on('click', function(){
      classification();
    })
    
  })
   // ///////////////////////////////////////////
   // 録音関係
   // ///////////////////////////////////////////
  
   // 変数定義
   let localMediaStream = null;
   let localScriptProcessor = null;
   let audioSampleRate = null;
   let audioContext = null;
   let bufferSize = 1024;
   let audioData = []; // 録音データ
   let recordingFlg = false;
   let dl = null;
   let blob = null;　
    
   // 録音バッファ作成（録音中自動で繰り返し呼び出される）
   function onAudioProcess(e) {
       if (!recordingFlg) return;
       console.log('onAudioProcess');
  
       // 音声のバッファを作成
       let input = e.inputBuffer.getChannelData(0);
       let bufferData = new Float32Array(bufferSize);
       for (let i = 0; i < bufferSize; i++) {
           bufferData[i] = input[i];
       }
       audioData.push(bufferData);
   }
  
   // 解析開始
   function startRecording(evt_stream) {
       // 画面アクセス時にマイクを取得
       console.log('startRecording');
       recordingFlg = true;
  
       // 取得されている音声ストリームの録音を開始
       localMediaStream = evt_stream;
  
       if (!navigator || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
         alert('Missing support for navigator.mediaDevices.getUserMedia') // temp: helps when testing for strange issues on ios/safari
         return
       }
  
       audioContext = new (window.AudioContext || window.webkitAudioContext)();
       // サンプルレートを保持しておく
       audioSampleRate = audioContext.sampleRate;
  
       let scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
       localScriptProcessor = scriptProcessor;
  
       if (audioContext.createMediaStreamDestination) {
         destinationNode = audioContext.createMediaStreamDestination()
       }
       else {
         destinationNode = audioContext.destination
       }
  
       // safariで Web Audio APIを動かすため、先にaudioContextを生成し、UserMediaを生成する
       return navigator.mediaDevices.getUserMedia({audio: true})
         .then((stream) => {
           this._startRecordingWithStream(stream, destinationNode, scriptProcessor)
         })
         .catch((error) => {
           alert('Error with getUserMedia: ' + error.message) // temp: helps when testing for strange issues on ios/safari
           console.log(error)
         })
     }
  
     function _startRecordingWithStream(stream, destinationNode, scriptProcessor) {
       // ループ処理のセット
       let mediastreamsource = audioContext.createMediaStreamSource(stream);
       mediastreamsource.connect(scriptProcessor);
       scriptProcessor.onaudioprocess = onAudioProcess;
       console.log('startRecording scriptProcessor.connect(audioContext.destination)');
       scriptProcessor.connect(destinationNode);
     }
  
   // 解析終了
   function endRecording() {
       console.log('endRecording');
       recordingFlg = false;
       // console.log('audioData');
       // console.log(audioData);
  
       // console.log('blob = exportWAV(audioData)');
       // 録音できたので録音データをwavにしてinputに配置＆再生ボタンに登録
       blob = exportWAV(audioData);
       // データ送信用のinputタグを取得
       let wave_tag = document.getElementById('demo_speaking_wave_file');
  
       // base64加工
       let reader = new FileReader();
       reader.readAsDataURL(blob);
       reader.onloadend = function() {
           base64data = reader.result;
           // console.log('base64data');
           // console.log(base64data);
          //wave_tag.value = base64data;
       };
  
       let myURL = window.URL || window.webkitURL;
       let url = myURL.createObjectURL(blob);
       //dl = document.querySelector("#dl");
       
       dl = "test";
       //集音したものから音声データを作成する
       //dl.href = myURL.createObjectURL(blob);
       
       //var min = 0 ;
  　　 //var max = 9 ;
  
       //var a = Math.floor( Math.random() * (max + 1 - min) ) + min ;
       //var b = Math.floor( Math.random() * (max + 1 - min) ) + min ;
       //dl.download = new Date().getTime().toString(16) + '-' + a + b + '.wav';
       // console.log('wavefile');
       // console.log(url);
  
       // audioタグに録音データをセット
       let player = document.getElementById('player');
       player.src =  url;
       player.load();
  
       // audioDataをクリア
       localMediaStream = null;
       localScriptProcessor = null;
       audioContext.close()
       audioContext = null;
       audioData = []; // 録音データ
   }
  
   // ///////////////////////////////////////////
   // waveファイル作成処理
   // ///////////////////////////////////////////
  
   function exportWAV(audioData) {
  
       let encodeWAV = function(samples, sampleRate) {
           let buffer = new ArrayBuffer(44 + samples.length * 2);
           let view = new DataView(buffer);
  
           let writeString = function(view, offset, string) {
               for (let i = 0; i < string.length; i++){
                   view.setUint8(offset + i, string.charCodeAt(i));
               }
           };
  
           let floatTo16BitPCM = function(output, offset, input) {
               for (let i = 0; i < input.length; i++, offset += 2){
                   let s = Math.max(-1, Math.min(1, input[i]));
                   output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
               }
           };
  
           writeString(view, 0, 'RIFF');  // RIFFヘッダ
           view.setUint32(4, 32 + samples.length * 2, true); // これ以降のファイルサイズ
           writeString(view, 8, 'WAVE'); // WAVEヘッダ
           writeString(view, 12, 'fmt '); // fmtチャンク
           view.setUint32(16, 16, true); // fmtチャンクのバイト数
           view.setUint16(20, 1, true); // フォーマットID
           view.setUint16(22, 1, true); // チャンネル数
           view.setUint32(24, sampleRate, true); // サンプリングレート
           view.setUint32(28, sampleRate * 2, true); // データ速度
           view.setUint16(32, 2, true); // ブロックサイズ
           view.setUint16(34, 16, true); // サンプルあたりのビット数
           writeString(view, 36, 'data'); // dataチャンク
           view.setUint32(40, samples.length * 2, true); // 波形データのバイト数
           floatTo16BitPCM(view, 44, samples); // 波形データ
  
           return view;
       };
  
       let mergeBuffers = function(audioData) {
           let sampleLength = 0;
           for (let i = 0; i < audioData.length; i++) {
               sampleLength += audioData[i].length;
           }
           let samples = new Float32Array(sampleLength);
           let sampleIdx = 0;
           for (let i = 0; i < audioData.length; i++) {
               for (let j = 0; j < audioData[i].length; j++) {
                   samples[sampleIdx] = audioData[i][j];
                   sampleIdx++;
               }
           }
           return samples;
       };
  
       let dataview = encodeWAV(mergeBuffers(audioData), audioSampleRate);
       let audioBlob = new Blob([dataview], { type: 'audio/wav' });
  
       return audioBlob;
  
       // let myURL = window.URL || window.webkitURL;
       // let url = myURL.createObjectURL(audioBlob);
       // return url;
   }
  
   function audioPlay() {
       let play_button = document.getElementById("btn_play_pause");
       play_button.onclick = new Function("audioPause();");
       play_button.innerText = "停止";
       document.getElementById("player").play();
   }
  
   function audioPause() {
       let play_button = document.getElementById("btn_play_pause");
       play_button.onclick = new Function("audioPlay();");
       play_button.innerText = "再生";
       document.getElementById("player").pause();
   }
      
    function classification() {
        
        if (dl === null || dl === undefined) {
           document.getElementById("result").innerText = "打音が録音されてません" ;  
        } else {　
            // POSTでアップロード
        $.ajax({
            url  : "./classification",
            type : "POST",
            data : blob,
            cache       : false,
            contentType : false,
            processData : false,
            dataType    : "html"
        })
        .done(function(data, textStatus, jqXHR) {
          var result_element = document.getElementById('result');
  
          var objResult = JSON.parse(data);
          console.log(objResult);
  
          // 子ノードを全削除
          if (result_element.hasChildNodes()){
            for (var i=result_element.childNodes.length-1; i>=0; i--) {
              result_element.removeChild(result_element.childNodes[i]);
            }
          }
  
          // classifierId+name
          var classifier;
          classifier = document.createTextNode("classifierId="+objResult.images[0].classifiers[0].classifier_id+", classifierName="+objResult.images[0].classifiers[0].name);
          var classifierBox = document.createElement('p');
          classifierBox.appendChild(classifier);
          result_element.appendChild(classifierBox);
  
          // classes
          var classes;
          classes = document.createTextNode("className="+objResult.images[0].classifiers[0].classes[0].class +", score="+objResult.images[0].classifiers[0].classes[0].score +", className="+objResult.images[0].classifiers[0].classes[1].class +", score="+objResult.images[0].classifiers[0].classes[1].score );
          var classesBox = document.createElement('p');
          classesBox.appendChild(classes);
          result_element.appendChild(classesBox);
        })
        .fail(function(jqXHR, textStatus, errorThrown) {
            // エラー
            var error;
            error = document.createTextNode("error="+data);
            var errorBox = document.createElement('p');
            errorBox.appendChild(error);
            result_element.appendChild(errorBox);
            return;
        })
        .always(function(data) {
          // do nothing
        });
      }
    }
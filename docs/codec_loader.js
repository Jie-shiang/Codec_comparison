/**
 * Codec Loader - Shared codec data loading module
 * Dynamically loads codec configurations and audio file data
 */

class CodecLoader {
    constructor(configPath, audioBasePath, configBasePath) {
        this.configPath = configPath;
        this.audioBasePath = audioBasePath;
        this.configBasePath = configBasePath;
        this.codecsConfig = null;
        
        // Initialize datasets structure
        this.datasets = {
            librispeech: {
                files: ['Total', 'Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5', 'Error_Sample_1'],
                errorExamples: ['Error_Sample_1'],
                codecData: {}
            },
            commonvoice: {
                files: ['Total', 'Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5', 'Error_Sample_1'],
                errorExamples: ['Error_Sample_1'],
                codecData: {}
            }
        };
    }

    /**
     * Load codecs configuration file
     */
    async loadCodecsConfig() {
        try {
            const response = await fetch(this.configPath);
            if (!response.ok) {
                throw new Error(`Failed to load config: ${response.status}`);
            }
            this.codecsConfig = await response.json();
            return this.codecsConfig;
        } catch (error) {
            console.error('Error loading codecs config:', error);
            return null;
        }
    }

    /**
     * Load specific codec data
     */
    async loadCodecData(dataset, codecId, config) {
        const jsonPath = `${this.configBasePath}/${codecId}_${config}_config.json`;
        
        try {
            const response = await fetch(jsonPath);
            if (response.ok) {
                const data = await response.json();
                
                // Initialize codec data structure
                if (!this.datasets[dataset].codecData[codecId]) {
                    this.datasets[dataset].codecData[codecId] = {};
                }
                
                this.datasets[dataset].codecData[codecId][config] = data;
                return data;
            } else {
                console.warn(`Failed to load ${jsonPath}: ${response.status}`);
                return null;
            }
        } catch (error) {
            console.error(`Error loading ${jsonPath}:`, error);
            return null;
        }
    }

    /**
     * Load all codec data
     */
    async loadAllCodecData(codecsConfig) {
        const loadPromises = [];
        const datasetNames = Object.keys(this.datasets);
        
        for (const datasetName of datasetNames) {
            for (const [codecId, codecInfo] of Object.entries(codecsConfig)) {
                for (const config of codecInfo.configurations) {
                    loadPromises.push(
                        this.loadCodecData(datasetName, codecId, config)
                    );
                }
            }
        }
        
        await Promise.all(loadPromises);
        return true;
    }

    /**
     * Get specific codec data
     */
    getCodecData(dataset, codecId, config, fileId, dataType) {
        try {
            const codecData = this.datasets[dataset].codecData[codecId]?.[config];
            if (!codecData) return null;
            
            const datasetKey = this.getDatasetKey(dataset);
            const fileData = codecData[datasetKey]?.[fileId];
            
            if (!fileData) return null;
            
            switch (dataType) {
                case 'transcription':
                    return fileData.Transcription || null;
                case 'origin':
                    return fileData.Origin || null;
                case 'inference':
                    return fileData.Inference || null;
                case 'utmos':
                    return fileData.UTMOS || null;
                case 'dwer':
                    return fileData.dWER || fileData.dCER || null;
                case 'dwer_type':
                    if (fileData.dWER !== undefined) return 'dWER';
                    if (fileData.dCER !== undefined) return 'dCER';
                    return null;
                case 'pesq':
                    return fileData.PESQ || null;
                case 'stoi':
                    return fileData.STOI || null;
                case 'speaker_sim':
                    return fileData.Speaker_Sim || null;
                case 'filename':
                    return fileData.File_name || null;
                default:
                    return null;
            }
        } catch (error) {
            console.error('Error getting codec data:', error);
            return null;
        }
    }

    /**
     * Get audio file path
     */
    getAudioPath(dataset, codecId, config, filename) {
        if (!filename || filename === 'Total') {
            return '';
        }
        
        const datasetPath = dataset === 'librispeech' ? 'LibriSpeech' : 'CommonVoice';
        
        if (codecId === 'original') {
            // Check if this is noise experiment (audio_noise base path)
            if (this.audioBasePath.includes('noise')) {
                return `${this.audioBasePath}/${datasetPath}/Noise/original/${filename}.flac`;
            } else {
                return `${this.audioBasePath}/${datasetPath}/original/${filename}.flac`;
            }
        } else {
            // Check if this is noise experiment
            if (this.audioBasePath.includes('noise')) {
                return `${this.audioBasePath}/${datasetPath}/Noise/${codecId}/${config}/${filename}.wav`;
            } else {
                return `${this.audioBasePath}/${datasetPath}/${codecId}/${config}/${filename}.wav`;
            }
        }
    }

    /**
     * Convert dataset key to corresponding key in JSON
     */
    getDatasetKey(dataset) {
        switch(dataset) {
            case 'librispeech':
                // Check if using noise configs
                if (this.configBasePath.includes('noise')) {
                    return 'LibriSpeech_Noise';
                }
                return 'LibriSpeech';
            case 'commonvoice':
                // Check if using noise configs
                if (this.configBasePath.includes('noise')) {
                    return 'CommonVoice_Noise';
                }
                return 'CommonVoice';
            default:
                return 'LibriSpeech';
        }
    }

    /**
     * Get all datasets
     */
    getDatasets() {
        return this.datasets;
    }
}
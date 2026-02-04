#!/usr/bin/env python3
"""
패션 JSON 인코더 - 메인 통합 스크립트

이 스크립트는 데이터 처리, 모델 학습, 평가를 포함한 모든 구성 요소를 통합하여
패션 JSON 인코더 시스템의 메인 진입점을 제공합니다.

사용법:
    python main.py --help                          # 도움말 표시
    python main.py train --dataset_path /path/to/data  # 모델 학습
    python main.py evaluate --checkpoint_path /path/to/checkpoint  # 모델 평가
    python main.py sanity_check                    # 정상성 검사 실행
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from data.fashion_dataset import FashionDataModule
from training.trainer import FashionTrainer, create_trainer_from_data_module
from utils.config import TrainingConfig
from utils.validators import InputValidator, ModelValidator
from examples.json_encoder_sanity_check import JSONEncoderSanityChecker


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_encoder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FashionEncoderSystem:
    """
    모든 패션 JSON 인코더 구성 요소를 통합하는 메인 시스템 클래스입니다.
    
    이 클래스는 패션 JSON 인코더 시스템의 학습, 평가, 추론을 위한
    통합된 인터페이스를 제공합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        패션 인코더 시스템을 초기화합니다.
        
        Args:
            config_path: 설정 파일 경로 (선택사항)
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.data_module = None
        self.trainer = None
        
        logger.info(f"패션 인코더 시스템이 초기화되었습니다")
        logger.info(f"장치: {self.device}")
        logger.info(f"설정: {self.config}")
    
    def _load_config(self, config_path: Optional[str]) -> TrainingConfig:
        """파일에서 설정을 로드하거나 기본값을 사용합니다."""
        if config_path and Path(config_path).exists():
            logger.info(f"{config_path}에서 설정을 로드합니다")
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # dict를 TrainingConfig로 변환
            config = TrainingConfig(**config_dict)
        else:
            logger.info("기본 설정을 사용합니다")
            config = TrainingConfig()
        
        return config
    
    def _setup_device(self) -> str:
        """컴퓨팅 장치를 설정하고 검증합니다."""
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"CUDA 사용 가능: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = 'cpu'
            logger.warning("CUDA를 사용할 수 없어 CPU를 사용합니다")
        
        return device
    def create_config_file(output_path: str, auto_categories: list = None) -> None:
        """샘플 설정 파일을 생성합니다. 스타일 기반 target_categories 자동 적용 가능"""
        
        # 기본 config 생성
        config = TrainingConfig()
        
        # target_categories 설정
        if auto_categories is not None and len(auto_categories) > 0:
            config.target_categories = auto_categories
        else:
            # 기본값
            config.target_categories = ["레트로", "로맨틱", "리조트"]

        # dict로 변환
        config_dict = {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'temperature': config.temperature,
            'embedding_dim': config.embedding_dim,
            'hidden_dim': config.hidden_dim,
            'output_dim': config.output_dim,
            'dropout_rate': config.dropout_rate,
            'weight_decay': config.weight_decay,
            'max_epochs': config.max_epochs,
            'target_categories': config.target_categories,
            'image_size': config.image_size,
            'crop_padding': config.crop_padding
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"샘플 설정 파일이 생성되었습니다: {output_path}")

    
    
    def setup_data(self, dataset_path: str, **kwargs) -> None:
        logger.info(f"데이터셋으로 데이터 모듈을 설정합니다: {dataset_path}")
        
        # JSON 폴더 확인
        annotations_dir = Path(dataset_path) / "annotations"
        if not annotations_dir.exists():
            logger.warning("JSON 디렉토리를 찾을 수 없습니다. 기본 target_categories 사용")
            detected_categories = self.config.target_categories
        else:
            # JSON 파일에서 스타일 읽기
            detected_categories = set()
            for json_file in annotations_dir.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labels = data.get("라벨링", {})
                    styles = labels.get("스타일", [])
                    for style_entry in styles:
                        style_name = style_entry.get("스타일")
                        if style_name:
                            detected_categories.add(style_name)
            detected_categories = list(detected_categories)
            if len(detected_categories) == 0:
                logger.warning("스타일 정보가 JSON에서 발견되지 않았습니다. 기본 target_categories 사용")
                detected_categories = self.config.target_categories

        logger.info(f"자동 감지된 target_categories: {detected_categories}")
        
        # data_module 생성
        data_config = {
            'dataset_path': dataset_path,
            'target_categories': detected_categories,
            'batch_size': self.config.batch_size,
            'image_size': self.config.image_size,
            **kwargs
        }
        
        self.data_module = FashionDataModule(**data_config)
        
        try:
            self.data_module.setup()
            vocab_sizes = self.data_module.get_vocab_sizes()
            logger.info(f"데이터 설정이 성공적으로 완료되었습니다")
            logger.info(f"학습 샘플: {len(self.data_module.train_dataset)}")
            logger.info(f"검증 샘플: {len(self.data_module.val_dataset)}")
            logger.info(f"어휘 크기: {vocab_sizes}")
        except Exception as e:
            logger.error(f"데이터 설정 실패: {e}")
            raise


    
    def setup_trainer(self, checkpoint_dir: str = 'checkpoints', 
                     log_dir: str = 'logs') -> None:
        """
        설정된 데이터 모듈로 트레이너를 설정합니다.
        
        Args:
            checkpoint_dir: 모델 체크포인트를 저장할 디렉토리
            log_dir: 학습 로그를 저장할 디렉토리
        """
        if self.data_module is None:
            raise ValueError("데이터 모듈이 초기화되지 않았습니다. 먼저 setup_data()를 호출하세요.")
        
        logger.info(f"트레이너를 설정합니다")
        logger.info(f"체크포인트 디렉토리: {checkpoint_dir}")
        logger.info(f"로그 디렉토리: {log_dir}")
        
        self.trainer = create_trainer_from_data_module(
            data_module=self.data_module,
            config=self.config,
            device=self.device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )
        
        logger.info("트레이너 설정이 완료되었습니다")
    
    def train(self, standalone_epochs: int = 5, 
              contrastive_epochs: Optional[int] = None,
              save_results: bool = True) -> Dict[str, Any]:
        """
        완전한 학습 파이프라인을 실행합니다.
        
        Args:
            standalone_epochs: 독립 JSON 인코더 학습을 위한 에포크 수
            contrastive_epochs: 대조 학습을 위한 에포크 수 (None이면 config 사용)
            save_results: 학습 결과를 파일에 저장할지 여부
            
        Returns:
            학습 결과를 포함하는 딕셔너리
        """
        if self.trainer is None:
            raise ValueError("트레이너가 초기화되지 않았습니다. 먼저 setup_trainer()를 호출하세요.")
        
        logger.info("완전한 학습 파이프라인을 시작합니다")
        
        # 데이터 로더 가져오기
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        results = {}
        
        # 1단계: 독립 JSON 인코더 학습
        if standalone_epochs > 0:
            logger.info(f"1단계: 독립 JSON 인코더 학습 ({standalone_epochs} 에포크)")
            
            standalone_results = self.trainer.train_json_encoder_standalone(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=standalone_epochs
            )
            
            results['standalone'] = standalone_results
            logger.info(f"독립 학습이 완료되었습니다. 최종 손실: {standalone_results['val_losses'][-1]:.4f}")
        
        # 2단계: 대조 학습
        if contrastive_epochs is None:
            contrastive_epochs = self.config.max_epochs
        
        if contrastive_epochs > 0:
            logger.info(f"2단계: 대조 학습 ({contrastive_epochs} 에포크)")
            
            contrastive_results = self.trainer.train_contrastive_learning(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=contrastive_epochs
            )
            
            results['contrastive'] = contrastive_results
            logger.info(f"대조 학습이 완료되었습니다. 최고 손실: {contrastive_results['best_val_loss']:.4f}")
        
        # 결과 저장
        if save_results:
            self._save_training_results(results)
        
        logger.info("완전한 학습 파이프라인이 완료되었습니다")
        return results
    
    def evaluate(self, checkpoint_path: str, 
                 dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        학습된 모델을 평가합니다.
        
        Args:
            checkpoint_path: 모델 체크포인트 경로
            dataset_path: 평가 데이터셋 경로 (None이면 학습 데이터 사용)
            
        Returns:
            평가 결과를 포함하는 딕셔너리
        """
        logger.info(f"체크포인트에서 모델을 평가합니다: {checkpoint_path}")
        
        # 아직 설정되지 않았다면 데이터 설정
        if self.data_module is None and dataset_path:
            self.setup_data(dataset_path)
        elif self.data_module is None:
            raise ValueError("사용 가능한 데이터 모듈이 없습니다. dataset_path를 제공하거나 먼저 setup_data()를 호출하세요.")
        
        # 아직 설정되지 않았다면 트레이너 설정
        if self.trainer is None:
            self.setup_trainer()
        
        # 체크포인트 로드
        checkpoint = self.trainer.load_checkpoint(checkpoint_path)
        
        # 평가 실행
        val_loader = self.data_module.val_dataloader()
        evaluation_results = self.trainer._final_evaluation(val_loader)
        
        logger.info(f"평가가 완료되었습니다")
        logger.info(f"결과: {evaluation_results}")
        
        return evaluation_results
    
    def sanity_check(self, dataset_path: Optional[str] = None,
                    num_epochs: int = 3) -> Dict[str, Any]:
        """
        시스템의 포괄적인 정상성 검사를 실행합니다.
        
        Args:
            dataset_path: 데이터셋 경로 (None이면 합성 데이터 사용)
            num_epochs: 정상성 검사 학습을 위한 에포크 수
            
        Returns:
            정상성 검사 결과를 포함하는 딕셔너리
        """
        logger.info("시스템 정상성 검사를 실행합니다")
        
        try:
            # 데이터 설정
            if dataset_path:
                self.setup_data(dataset_path, batch_size=16, num_workers=0)
                vocab_sizes = self.data_module.get_vocab_sizes()
            else:
                # 합성 데이터 사용
                logger.info("정상성 검사를 위해 합성 데이터를 사용합니다")
                vocab_sizes = {
                    'category': 10,
                    'style': 20,
                    'silhouette': 15,
                    'material': 25,
                    'detail': 30
                }
                self.data_module = self._create_synthetic_data_module(vocab_sizes)
            
            # 정상성 검사 실행
            checker = JSONEncoderSanityChecker(vocab_sizes, self.device)
            results = checker.run_sanity_check(self.data_module, num_epochs)
            
            # 결과 저장
            self._save_sanity_check_results(results)
            
            logger.info("정상성 검사가 완료되었습니다")
            return results
            
        except Exception as e:
            logger.error(f"정상성 검사 실패: {e}")
            raise
    
    def _create_synthetic_data_module(self, vocab_sizes: Dict[str, int]):
        """테스트를 위한 합성 데이터 모듈을 생성합니다."""
        from examples.json_encoder_sanity_check import create_synthetic_data_module
        return create_synthetic_data_module(vocab_sizes, self.device)
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """학습 결과를 파일에 저장합니다."""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / "training_results.json"
        
        # JSON 직렬화 가능한 형식으로 변환
        json_results = self._convert_to_json_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"학습 결과가 저장되었습니다: {results_file}")
    
    def _save_sanity_check_results(self, results: Dict[str, Any]) -> None:
        """정상성 검사 결과를 파일에 저장합니다."""
        output_dir = Path("temp_logs")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / "sanity_check_results.json"
        
        # JSON 직렬화 가능한 형식으로 변환
        json_results = self._convert_to_json_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"정상성 검사 결과가 저장되었습니다: {results_file}")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """객체를 JSON 직렬화 가능한 형식으로 변환합니다."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() <= 1000 else f"<Tensor shape={list(obj.shape)}>"
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json_serializable(obj.__dict__)
        else:
            return obj
    
    def cleanup(self) -> None:
        """시스템 리소스를 정리합니다."""
        if self.trainer:
            self.trainer.close()
        
        logger.info("시스템 정리가 완료되었습니다")


def create_config_file(output_path: str) -> None:
    """샘플 설정 파일을 생성합니다."""
    config = TrainingConfig()
    config.target_categories = ["레트로", "로맨틱", "리조트"]
    
    config_dict = {
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'temperature': config.temperature,
        'embedding_dim': config.embedding_dim,
        'hidden_dim': config.hidden_dim,
        'output_dim': config.output_dim,
        'dropout_rate': config.dropout_rate,
        'weight_decay': config.weight_decay,
        'max_epochs': config.max_epochs,
        'target_categories': config.target_categories,
        'image_size': config.image_size,
        'crop_padding': config.crop_padding
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"샘플 설정 파일이 생성되었습니다: {output_path}")


def main():
    """명령줄 인터페이스가 있는 메인 진입점입니다."""
    parser = argparse.ArgumentParser(
        description="패션 JSON 인코더 - 메인 통합 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py train --dataset_path /path/to/kfashion
  python main.py train --dataset_path /path/to/kfashion --config config.json
  python main.py evaluate --checkpoint_path checkpoints/best_model.pt
  python main.py sanity_check
  python main.py sanity_check --dataset_path /path/to/kfashion
  python main.py create_config --output config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 학습 명령어
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--dataset_path', required=True, 
                             help='K-Fashion 데이터셋 경로')
    train_parser.add_argument('--config', 
                             help='설정 파일 경로')
    train_parser.add_argument('--checkpoint_dir', default='checkpoints',
                             help='체크포인트를 저장할 디렉토리')
    train_parser.add_argument('--log_dir', default='logs',
                             help='로그를 저장할 디렉토리')
    train_parser.add_argument('--standalone_epochs', type=int, default=5,
                             help='독립 학습을 위한 에포크 수')
    train_parser.add_argument('--contrastive_epochs', type=int,
                             help='대조 학습을 위한 에포크 수')
    train_parser.add_argument('--batch_size', type=int,
                             help='배치 크기 (설정 재정의)')
    train_parser.add_argument('--learning_rate', type=float,
                             help='학습률 (설정 재정의)')
    
    # 평가 명령어
    eval_parser = subparsers.add_parser('evaluate', help='학습된 모델 평가')
    eval_parser.add_argument('--checkpoint_path', required=True,
                            help='모델 체크포인트 경로')
    eval_parser.add_argument('--dataset_path',
                            help='평가 데이터셋 경로')
    eval_parser.add_argument('--config',
                            help='설정 파일 경로')
    
    # 정상성 검사 명령어
    sanity_parser = subparsers.add_parser('sanity_check', help='시스템 정상성 검사 실행')
    sanity_parser.add_argument('--dataset_path',
                              help='데이터셋 경로 (제공되지 않으면 합성 데이터 사용)')
    sanity_parser.add_argument('--config',
                              help='설정 파일 경로')
    sanity_parser.add_argument('--epochs', type=int, default=3,
                              help='정상성 검사를 위한 에포크 수')
    
    # 설정 생성 명령어
    config_parser = subparsers.add_parser('create_config', help='샘플 설정 파일 생성')
    config_parser.add_argument('--output', default='config.json',
                              help='설정 파일 출력 경로')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # create_config 명령어 처리
    if args.command == 'create_config':
        create_config_file(args.output)
        return
    
    # 시스템 초기화
    try:
        system = FashionEncoderSystem(config_path=getattr(args, 'config', None))
        
        # 명령줄 인수로 설정 재정의
        if hasattr(args, 'batch_size') and args.batch_size:
            system.config.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            system.config.learning_rate = args.learning_rate
        
        # 명령어 실행
        if args.command == 'train':
            system.setup_data(args.dataset_path)
            system.setup_trainer(args.checkpoint_dir, args.log_dir)
            
            contrastive_epochs = args.contrastive_epochs
            if contrastive_epochs is None:
                contrastive_epochs = system.config.max_epochs
            
            results = system.train(
                standalone_epochs=args.standalone_epochs,
                contrastive_epochs=contrastive_epochs
            )
            
            print("\n" + "="*60)
            print("학습이 성공적으로 완료되었습니다")
            print("="*60)
            if 'standalone' in results:
                print(f"독립 학습 최종 손실: {results['standalone']['val_losses'][-1]:.4f}")
            if 'contrastive' in results:
                print(f"대조 학습 최고 손실: {results['contrastive']['best_val_loss']:.4f}")
                print(f"최종 메트릭: {results['contrastive']['final_metrics']}")
        
        elif args.command == 'evaluate':
            results = system.evaluate(
                checkpoint_path=args.checkpoint_path,
                dataset_path=getattr(args, 'dataset_path', None)
            )
            
            print("\n" + "="*60)
            print("평가가 완료되었습니다")
            print("="*60)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        elif args.command == 'sanity_check':
            results = system.sanity_check(
                dataset_path=getattr(args, 'dataset_path', None),
                num_epochs=args.epochs
            )
            
            print("\n" + "="*60)
            print("정상성 검사가 완료되었습니다")
            print("="*60)
            
            validation = results.get('validation_results', {})
            all_passed = all(validation.get(check, False) for check in validation if check != 'errors')
            
            if all_passed:
                print("🎉 모든 정상성 검사가 통과했습니다!")
            else:
                print("⚠️  일부 정상성 검사가 실패했습니다. 자세한 내용은 로그를 확인하세요.")
        
        # 정리
        system.cleanup()
        
    except Exception as e:
        logger.error(f"명령어 실행 실패: {e}")
        print(f"오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
import os
import json
import torch
import chess
import berserk
import boto3
from google.cloud import storage
from azure.storage.blob import BlobServiceClient
import paramiko
from src.training.reinforcement.mcts import MCTS

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def load_opening_book(opening_book_path):
    with open(opening_book_path, 'r') as f:
        book_data = json.load(f)
    return book_data

def choose_opening_move(board, opening_book):
    fen = board.fen()
    if fen in opening_book:
        moves = opening_book[fen]
        if moves:
            return chess.Move.from_uci(moves[0])
    return None

class LichessBotDeploymentWorker:
    def __init__(self, model_path, opening_book_path, lichess_token, time_control, rating_range, use_mcts, cloud_provider, progress_callback, status_callback):
        self.model_path = model_path
        self.opening_book_path = opening_book_path
        self.lichess_token = lichess_token
        self.time_control = time_control
        self.rating_range = rating_range
        self.use_mcts = use_mcts
        self.cloud_provider = cloud_provider
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.model = None
        self.opening_book = None
        self.mcts = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._compile_chess_engine()

    def _compile_chess_engine(self):
        self.status_callback("Loading model...")
        self.model = load_model(self.model_path, self.device)
        self.progress_callback(25)
        self.status_callback("Loading opening book...")
        self.opening_book = load_opening_book(self.opening_book_path)
        self.progress_callback(50)
        if self.use_mcts:
            self.status_callback("Initializing MCTS...")
            self.mcts = MCTS(model=self.model, device=self.device, c_puct=1.4, n_simulations=800)
        else:
            self.mcts = None
        self.progress_callback(75)
        self.status_callback("Chess engine compiled successfully.")
        self.progress_callback(100)

    def deploy_to_cloud(self):
        self.status_callback(f"Deploying to {self.cloud_provider}...")
        self.progress_callback(10)
        provider = self.cloud_provider.strip().lower()
        if provider == "aws":
            self._deploy_aws()
        elif provider == "google cloud":
            self._deploy_gcp()
        elif provider == "azure":
            self._deploy_azure()
        elif provider == "other":
            self._deploy_other()
        else:
            self.status_callback("Cloud provider not recognized. Skipping deployment step.")
            self.progress_callback(100)
            return
        self.progress_callback(100)
        self.status_callback("Deployment complete.")

    def _deploy_aws(self):
        try:
            self.status_callback("Connecting to AWS...")
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region_name=os.environ.get("AWS_DEFAULT_REGION")
            )
            bucket_name = "my-chess-engine-bucket"
            model_s3_key = os.path.basename(self.model_path)
            book_s3_key = os.path.basename(self.opening_book_path)
            self.status_callback("Uploading model to S3...")
            s3_client.upload_file(self.model_path, bucket_name, model_s3_key)
            self.progress_callback(50)
            self.status_callback("Uploading opening book to S3...")
            s3_client.upload_file(self.opening_book_path, bucket_name, book_s3_key)
            self.progress_callback(90)
        except Exception as e:
            self.status_callback(f"AWS deployment failed: {e}")
            self.progress_callback(100)
            raise

    def _deploy_gcp(self):
        try:
            self.status_callback("Connecting to GCP...")
            gcp_storage_client = storage.Client()
            bucket_name = "my-chess-engine-bucket"
            bucket = gcp_storage_client.get_bucket(bucket_name)
            model_s3_key = os.path.basename(self.model_path)
            book_s3_key = os.path.basename(self.opening_book_path)
            self.status_callback("Uploading model to GCP bucket...")
            model_blob = bucket.blob(model_s3_key)
            model_blob.upload_from_filename(self.model_path)
            self.progress_callback(50)
            self.status_callback("Uploading opening book to GCP bucket...")
            book_blob = bucket.blob(book_s3_key)
            book_blob.upload_from_filename(self.opening_book_path)
            self.progress_callback(90)
        except Exception as e:
            self.status_callback(f"GCP deployment failed: {e}")
            self.progress_callback(100)
            raise

    def _deploy_azure(self):
        try:
            self.status_callback("Connecting to Azure...")
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING env variable is not set.")
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_name = "my-chess-engine-container"
            model_blob_name = os.path.basename(self.model_path)
            book_blob_name = os.path.basename(self.opening_book_path)
            container_client = blob_service_client.get_container_client(container_name)
            self.status_callback("Uploading model to Azure Blob Storage...")
            with open(self.model_path, "rb") as data:
                container_client.upload_blob(name=model_blob_name, data=data, overwrite=True)
            self.progress_callback(50)
            self.status_callback("Uploading opening book to Azure Blob Storage...")
            with open(self.opening_book_path, "rb") as data:
                container_client.upload_blob(name=book_blob_name, data=data, overwrite=True)
            self.progress_callback(90)
        except Exception as e:
            self.status_callback(f"Azure deployment failed: {e}")
            self.progress_callback(100)
            raise

    def _deploy_other(self):
        try:
            self.status_callback("Performing custom or self-hosted deployment steps...")
            hostname = os.environ.get("OTHER_HOSTNAME", "example.com")
            port = int(os.environ.get("OTHER_SSH_PORT", 22))
            username = os.environ.get("OTHER_SSH_USERNAME", "root")
            password = os.environ.get("OTHER_SSH_PASSWORD", None)
            if not password:
                raise ValueError("Must set OTHER_SSH_PASSWORD in environment for custom deployment.")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=hostname, port=port, username=username, password=password)
            sftp = ssh.open_sftp()
            remote_path_for_model = f"/home/{username}/" + os.path.basename(self.model_path)
            remote_path_for_book = f"/home/{username}/" + os.path.basename(self.opening_book_path)
            self.status_callback("Uploading model via SFTP...")
            sftp.put(self.model_path, remote_path_for_model)
            self.progress_callback(50)
            self.status_callback("Uploading opening book via SFTP...")
            sftp.put(self.opening_book_path, remote_path_for_book)
            self.progress_callback(90)
            sftp.close()
            ssh.close()
        except Exception as e:
            self.status_callback(f"Custom/Other deployment failed: {e}")
            self.progress_callback(100)
            raise

    def start_bot(self):
        self.status_callback("Starting Lichess bot...")
        try:
            session = berserk.TokenSession(self.lichess_token)
            client = berserk.Client(session=session)
            for event in client.bots.stream_events():
                if event['type'] == 'challenge':
                    self._handle_challenge_event(client, event['challenge'])
                elif event['type'] == 'gameStart':
                    game_id = event['game']['id']
                    self._play_game(client, game_id)
        except Exception as e:
            self.status_callback(f"Error starting bot or streaming events: {e}")
            raise

    def _handle_challenge_event(self, client, challenge):
        challenger_rating = challenge['challenger']['rating']
        if self.rating_range[0] <= challenger_rating <= self.rating_range[1]:
            client.bots.accept_challenge(challenge['id'])
            self.status_callback(f"Accepted challenge from rating {challenger_rating}.")
        else:
            client.bots.decline_challenge(challenge['id'])
            self.status_callback(f"Declined challenge from rating {challenger_rating}.")

    def _play_game(self, client, game_id):
        self.status_callback(f"Playing game: {game_id}")
        board = chess.Board()
        for event in client.bots.stream_game_state(game_id):
            state_type = event['type']
            if state_type == 'gameFull':
                moves = event['state'].get('moves', '').split()
                for mv in moves:
                    board.push_uci(mv)
            elif state_type == 'gameState':
                moves = event.get('moves', '').split()
                board.clear()
                board.reset()
                for mv in moves:
                    board.push_uci(mv)
                if board.is_game_over():
                    self.status_callback(f"Game over: {board.result()} for game {game_id}")
                    break
                if (board.turn == chess.WHITE and event['white']['id'] == game_id) or \
                   (board.turn == chess.BLACK and event['black']['id'] == game_id):
                    self._make_move(client, game_id, board)
            elif state_type == 'chatLine':
                pass
            if board.is_game_over():
                self.status_callback(f"Game over: {board.result()} for game {game_id}")
                break

    def _make_move(self, client, game_id, board):
        move = choose_opening_move(board, self.opening_book)
        if move is not None:
            best_move = move
        elif self.mcts:
            self.mcts.set_root_node(board)
            move_probs = self.mcts.get_move_probs()
            if not move_probs:
                self.status_callback("No moves from MCTS. Aborting move.")
                return
            best_move = max(move_probs, key=move_probs.get)
        else:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                self.status_callback("No legal moves available. Aborting move.")
                return
            best_move = legal_moves[0]
        board.push(best_move)
        uci_move = best_move.uci()
        try:
            client.bots.make_move(game_id, uci_move)
        except Exception as e:
            self.status_callback(f"Failed to make move: {e}")
            raise

    def run_all(self):
        self.deploy_to_cloud()
        self.start_bot()
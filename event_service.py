import httpx
import asyncio
import logging

from py_near.account import Account
from config.settings import conf
from models.block_model import BlockModel
from models.event_progress import EventProgressModel
from models.transaction_model import TransactionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EventService:
    def __init__(self, check_interval: int):
        self.check_interval = check_interval  # in seconds
        self.initialize_config()

    def initialize_config(self):
        self.filter_sender = conf.EVENT_FILTER_SENDER
        self.filter_recipient = conf.EVENT_FILTER_RECIPIENT
        self.event_batch_blocks_count = conf.EVENT_BATCH_BLOCKS_COUNT
        self.notification_target = conf.EVENT_NOTIFICATION_TARGET
        self.near_rpc = (
            "https://rpc.testnet.near.org"
            if conf.EVENT_NEAR_ACCOUNT_ID.endswith(".testnet")
            else "https://rpc.mainnet.near.org"
        )

    async def start_listeners(self):
        await self.listen_transactions()

    async def listen_transactions(self):
        """
        Listen for new transactions and send notifications based on the filters.
        """
        while True:
            start_block, end_block = await self.get_start_end_block_heights()
            logger.info("Processing blocks: #%s - #%s", start_block, end_block)

            for block_height in range(start_block, end_block + 1):
                transactions = await self.filter_transactions(block_height)
                for transaction in transactions:
                    await self.handle_notification(transaction)

            event_progress = await EventProgressModel.first()
            if event_progress:
                event_progress.last_block_height = end_block
                await event_progress.save()

            await asyncio.sleep(self.check_interval)

    async def init_event_progress(self):
        """
        Initialize the event progress with the first block in the database.
        """
        first_block = await BlockModel.all().order_by('block_height').first()
        if not first_block:
            logger.error("No blocks in the database.")
            return

        init_block_height = first_block.block_height
        await EventProgressModel.create(
            init_block_height=init_block_height,
            last_block_height=init_block_height,
        )
        return init_block_height

    async def get_start_end_block_heights(self):
        """
        Get the starting and ending block heights for the current batch.
        """
        event_progress = await EventProgressModel.first()
        start_block = event_progress.last_block_height if event_progress else await self.init_event_progress()
        end_block = start_block + self.event_batch_blocks_count
        return start_block, end_block

    async def filter_transactions(self, block_height):
        """
        Filter transactions based on block height, sender, and recipient filters.
        """
        transactions_query = TransactionModel.filter(block_height=block_height)

        if self.filter_sender:
            transactions_query = transactions_query.filter(signer_id=self.filter_sender)
        if self.filter_recipient:
            transactions_query = transactions_query.filter(receiver_id=self.filter_recipient)

        return await transactions_query.all()

    async def handle_notification(self, transaction):
        """
        Called for each transaction that matches the filters
        and sends the transaction data to the notification target.
        """
        try:
            if self.notification_target.startswith("http"):
                await self._send_http_notification(self.notification_target, transaction.serialize())
            elif "|" in self.notification_target:
                await self._invoke_contract(transaction)
            else:
                logger.error("No valid event notification target specified.")
        except Exception as e:
            logger.error(f"Error handling notification: {e}")

    # Send transaction data to an HTTP endpoint
    @staticmethod
    async def _send_http_notification(url: str, data: dict):
        """
        Send transaction data to an HTTP endpoint.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data)
                if response.status_code == 200:
                    logger.info(f"Successfully sent transaction data to {url}")
                else:
                    logger.error(f"Failed to send data to {url}: {response.status_code} - {response.text}")
            except httpx.RequestError as e:
                logger.error(f"HTTP request failed: {e}")

    async def _invoke_contract(self, transaction):
        """
        Call a NEAR smart-contract method with transaction data.
        In out smart-contract example it just logs the transaction data and increases a counter.
        """
        address, method = self.notification_target.split("|", 1)
        tx_data = transaction.serialize()

        try:
            near_account = self.get_near_account()
            result = await near_account.function_call(address, method, {"tx_data": tx_data})
            logger.info(f"Successfully called smart contract method with result: {result}")
        except Exception as e:
            logger.error(f"Smart contract call failed: {e}")

    def get_near_account(self):
        """Initialize a NEAR account object."""
        if not conf.EVENT_NEAR_ACCOUNT_ID or not conf.EVENT_NEAR_ACCOUNT_PRIVATE_KEY:
            raise ValueError("Account ID and Private Key are required for contract invocation.")
        return Account(conf.EVENT_NEAR_ACCOUNT_ID, conf.EVENT_NEAR_ACCOUNT_PRIVATE_KEY, self.near_rpc)

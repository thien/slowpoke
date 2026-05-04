"""Tests for the Bot base class."""

import unittest

from slowpoke.agents.bot import Bot


class TestBot(unittest.TestCase):
    """Test the abstract Bot base class."""

    def test_bot_name(self):
        """Bot should have the correct class name."""
        bot = Bot()
        self.assertEqual(type(bot).__name__, "Bot")

    def test_move_function_raises_not_implemented(self):
        """move_function should raise NotImplementedError."""
        bot = Bot()
        with self.assertRaises(NotImplementedError):
            bot.move_function(None, 0)

    def test_move_function_error_message(self):
        """Error message should include the class name."""
        bot = Bot()
        with self.assertRaises(NotImplementedError) as ctx:
            bot.move_function(None, 0)
        self.assertIn("Bot", str(ctx.exception))

    def test_bot_is_instantiable(self):
        """Bot base class can be instantiated directly."""
        bot = Bot()
        self.assertIsNotNone(bot)

    def test_bot_has_no_extra_attributes(self):
        """Bot should have no attributes by default."""
        bot = Bot()
        self.assertEqual(len(bot.__dict__), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
